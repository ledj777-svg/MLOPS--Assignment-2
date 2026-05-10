import torch
import json
import os
import wandb
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast, Trainer, TrainingArguments
from sklearn.metrics import classification_report
from data import prepare_data
from utils import compute_metrics, id2label

MODEL_NAME = "distilbert-base-cased"
MAX_LENGTH = 512
OUTPUT_DIR = "./results"


def get_best_checkpoint(output_dir):
    """Automatically find the latest checkpoint"""
    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"Results folder '{output_dir}' not found!")

    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]

    if not checkpoints:
        print("No checkpoint folder found. Using main results folder.")
        return output_dir

    # Take the latest checkpoint
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
    checkpoint_path = os.path.join(output_dir, latest_checkpoint)
    print(f"Using latest checkpoint: {checkpoint_path}")
    return checkpoint_path


def main():
    wandb.init(
        project="mlops-assignment2",
        name="distilbert-evaluation",
        job_type="evaluation"
    )

    print("Loading test dataset...")
    _, test_dataset, tokenizer = prepare_data(sample_size=2000, test_size=0.2)

    print("Loading trained model...")
    model_path = get_best_checkpoint(OUTPUT_DIR)

    model = DistilBertForSequenceClassification.from_pretrained(
        model_path,
        num_labels=len(id2label),
        local_files_only=True
    ).to('cuda' if torch.cuda.is_available() else 'cpu')

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_eval_batch_size=32,
        report_to="wandb",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        # tokenizer=tokenizer   ← Removed this line (causing error)
    )

    print(" Running final evaluation on test set")
    eval_results = trainer.evaluate()

    print("\n  Evaluation Results:")
    print(f"Loss       : {eval_results['eval_loss']:.4f}")
    print(f"Accuracy   : {eval_results['eval_accuracy']:.4f}")
    print(f"F1 Score   : {eval_results['eval_f1']:.4f}")

    wandb.log({
        "final/loss": eval_results.get("eval_loss", None),
        "final/accuracy": eval_results.get("eval_accuracy", None),
        "final/f1": eval_results.get("eval_f1", None),
    })

    print(" Generating classification report")
    predictions = trainer.predict(test_dataset)
    preds = predictions.predictions.argmax(-1)
    labels = predictions.label_ids

    report = classification_report(
        labels,
        preds,
        target_names=list(id2label.values()),
        output_dict=True,
        zero_division=0
    )

    with open("eval_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("Saved eval_report.json")

    # Log artifact
    artifact = wandb.Artifact("eval-report", type="evaluation")
    artifact.add_file("eval_report.json")
    wandb.log_artifact(artifact)

    wandb.finish()
    print("\n Evaluation completed successfully!")


if __name__ == "__main__":
    main()