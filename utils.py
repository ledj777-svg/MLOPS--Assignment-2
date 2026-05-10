import torch
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
import json

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

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    return {"accuracy": acc, "f1": f1}