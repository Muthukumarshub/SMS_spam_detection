# 📲 SMS Spam Detection

## 🚀 Overview
This project is a machine learning-based SMS spam detection system that classifies messages as **spam** or **ham (not spam)**. It uses Natural Language Processing (NLP) techniques and machine learning models to filter unwanted messages effectively.

## 🏆 Features
- Pretrained **SMS Spam Detection** model hosted on Hugging Face.
- Uses **NLP techniques** for text processing.
- **Machine Learning models** like Naïve Bayes, SVM, or deep learning for classification.
- Real-time detection of spam messages.

## 🎯 Model Details
The trained model is available on Hugging Face:
**[Muthukumar045454/SMS_Spam_detection](https://huggingface.co/Muthukumar045454/SMS_Spam_detection)**

This model is trained on a labeled dataset and optimized for detecting spam messages with high accuracy.

## 🛠️ Installation & Usage

### Prerequisites
Ensure you have Python installed along with the required dependencies:
```bash
pip install -r requirements.txt
```

### Running the Model
You can use the Hugging Face model directly in Python:
```python
from transformers import pipeline

classifier = pipeline("text-classification", model="Muthukumar045454/SMS_Spam_detection")

message = "Congratulations! You've won a free iPhone. Click here to claim."
prediction = classifier(message)
print(prediction)  # Output: Spam or Ham
```

### Running Locally
1. Clone the repository:
   ```bash
   git clone https://github.com/muthukumarshub/SMS_Spam_Detection.git
   cd SMS_Spam_Detection
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the script:
   ```bash
   python main.py
   ```

## 🧪 Dataset
The model was trained on a **publicly available SMS Spam dataset**, preprocessed using NLP techniques like tokenization, stopword removal, and TF-IDF vectorization.

## 📊 Results & Performance
- Achieved **high accuracy and precision** in detecting spam.
- Successfully reduces false positives and false negatives.
- Can be deployed for real-time SMS filtering.

## 🤖 Future Enhancements
- Deploy as a **web API** for easy integration with applications.
- Improve performance using **deep learning models**.
- Expand dataset to include more spam variations.

## 🤝 Contributing
Feel free to **fork** this repository, raise issues, and contribute improvements!

## 📜 License
This project is licensed under the MIT License.
---
Give a ⭐ if you found this useful! 🚀

![image](https://github.com/user-attachments/assets/edcbc0f8-06a2-4a63-8b24-ef390afea4dd)
![image](https://github.com/user-attachments/assets/1fc1a4de-b3b5-4b21-a14c-da8c67ffde61)
