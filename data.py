import requests
import gzip
import json
import random
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset

# ====================== LABEL MAPPING ======================
id2label = {
    0: "poetry",
    1: "children",
    2: "comics_graphic",
    3: "fantasy_paranormal",
    4: "history_biography",
    5: "mystery_thriller_crime",
    6: "romance",
    7: "young_adult"
}

label2id = {v: k for k, v in id2label.items()}


# ====================== DATASET CLASS ======================
class GoodreadsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# ====================== DATA LOADING ======================
def load_reviews(url, head=10000, sample_size=800):
    reviews = []
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()

    with gzip.open(response.raw, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if head and i >= head:
                break
            try:
                d = json.loads(line.strip())
                if 'review_text' in d and d['review_text'].strip():
                    reviews.append(d['review_text'])
            except:
                continue

    if len(reviews) > sample_size:
        reviews = random.sample(reviews, sample_size)
    return reviews


# ====================== PREPARE DATA ======================
def prepare_data(sample_size=200, test_size=0.2, max_length=512):
    """Prepare train and test datasets"""

    genre_url_dict = {
        'poetry': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_poetry.json.gz',
        'children': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_children.json.gz',
        'comics_graphic': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_comics_graphic.json.gz',
        'fantasy_paranormal': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_fantasy_paranormal.json.gz',
        'history_biography': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_history_biography.json.gz',
        'mystery_thriller_crime': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_mystery_thriller_crime.json.gz',
        'romance': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_romance.json.gz',
        'young_adult': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_young_adult.json.gz'
    }

    print("📥 Downloading and loading reviews...")
    genre_reviews_dict = {}

    for genre, url in genre_url_dict.items():
        print(f"   → Loading {genre}...")
        genre_reviews_dict[genre] = load_reviews(url, head=10000, sample_size=sample_size)

    # Create texts and labels
    texts = []
    labels = []
    for genre, reviews in genre_reviews_dict.items():
        texts.extend(reviews)
        labels.extend([label2id[genre]] * len(reviews))

    print(f"Total samples loaded: {len(texts)}")

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=test_size, stratify=labels, random_state=42
    )

    from transformers import DistilBertTokenizerFast
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')

    print("🔤 Tokenizing...")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=max_length)

    train_dataset = GoodreadsDataset(train_encodings, train_labels)
    test_dataset = GoodreadsDataset(test_encodings, test_labels)

    return train_dataset, test_dataset, tokenizer