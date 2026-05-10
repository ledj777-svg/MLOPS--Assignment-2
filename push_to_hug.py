from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
from huggingface_hub import login, upload_folder
import os


REPO_ID = "JD45-d/distilbert-goodreads-genre-classifier"
HF_TOKEN =  "# Your token"

print("Logging into Hugging Face...")
login(token=HF_TOKEN)
print("Login successful!\n")


print(" Loading model and tokenizer...")

model_path = "./results/checkpoint-640"

model = DistilBertForSequenceClassification.from_pretrained(model_path, local_files_only=True)
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-cased")


save_path = "./model_to_push"
os.makedirs(save_path, exist_ok=True)

model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print(f"Model and Tokenizer saved to {save_path}\n")


print(f"Uploading everything to {REPO_ID} ... (this may take some time)")

upload_folder(
    folder_path=save_path,
    repo_id=REPO_ID,
    repo_type="model",
    commit_message="Upload model and tokenizer",
    token=HF_TOKEN
)

print("\n SUCCESS! Model + Tokenizer should now be uploaded.")
print(f" Check here: https://huggingface.co/{REPO_ID}")
