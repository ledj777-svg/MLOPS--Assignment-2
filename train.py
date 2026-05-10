import torch
import wandb
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast, TrainingArguments, Trainer
from data import prepare_data
from utils import compute_metrics, id2label


model_name = 'distilbert-base-cased'
max_length = 512
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Using device: {device}")



wandb.init(
    project="mlops-assignment2",
    name="distilbert-run-1",
    config={
        "model": model_name,
        "epochs": 3,
        "batch_size": 16,
        "learning_rate": 3e-5,
        "max_length": max_length,
        "dataset": "UCSD Goodreads Reviews"
    }
)


print("Loading Tokenizer and Model from Hugging Face...")


tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)


model = DistilBertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(id2label)
).to(device)

print("Tokenizer and Model loaded successfully!")


train_dataset, test_dataset, _ = prepare_data()   # tokenizer already loaded above

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    warmup_steps=100,
    weight_decay=0.01,
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="wandb",
    run_name="distilbert-run-1",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

print("Starting Training")
trainer.train()

wandb.finish()