import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load Model
model = AutoModelForSequenceClassification.from_pretrained("model/spam_detector_model")
tokenizer = AutoTokenizer.from_pretrained("model/spam_detector_model")

# Function to Predict Spam
def predict_spam(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        output = model(**inputs)
    prediction = torch.argmax(output.logits, dim=1).item()
    return "Spam" if prediction == 1 else "Not Spam"

# Test Example
text = "You got 1000000000 in lottery"
print(f"Prediction: {predict_spam(text)}")
