import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, classification_report

# Load Trained Model & Tokenizer
model_path = "model/spam_detector_model"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Tokenize Test Data
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/spam.csv", encoding="latin-1")[["v1", "v2"]]
df.columns = ["label", "text"]
df["label"] = df["label"].map({"ham": 0, "spam": 1})

_, test_texts, _, test_labels = train_test_split(df["text"], df["label"], test_size=0.2)

test_encodings = tokenizer(list(test_texts), truncation=True, padding=True, max_length=512, return_tensors="pt")

# Perform Inference
model.eval()
with torch.no_grad():
    outputs = model(**test_encodings)
    predictions = torch.argmax(outputs.logits, dim=1)

# Evaluation
accuracy = accuracy_score(test_labels, predictions.numpy())
print(f"Accuracy: {accuracy:.4f}")
print(classification_report(test_labels, predictions.numpy()))
