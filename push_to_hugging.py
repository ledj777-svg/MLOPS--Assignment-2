from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
from huggingface_hub import login
import os


REPO_ID = "JD45-d/distilbert-goodreads-genre-classifier"

print(" Logging into Hugging Face...")

# Paste your latest token here (between the quotes)
HF_TOKEN = "your token"
login(token=HF_TOKEN)

print("Login successful!")



from huggingface_hub import create_repo
create_repo(REPO_ID, private=False, exist_ok=True)
print(f"Repository ready: {REPO_ID}")


print(" Loading model from checkpoint-640...")
model_path = "./results/checkpoint-640"

model = DistilBertForSequenceClassification.from_pretrained(
    model_path,
    local_files_only=True
)

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-cased")

print(f" Pushing model to {REPO_ID} ... This may take a few minutes.")

model.push_to_hub(REPO_ID)
tokenizer.push_to_hub(REPO_ID)

print("SUCCESS! Model pushed successfully!")
print(f" Link: https://huggingface.co/{REPO_ID}")
