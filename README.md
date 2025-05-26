🧠 Intelligent Fake News Detection using Transformer Networks
📌 Overview
This project implements a fake news classification system using a fine-tuned Transformer model (DistilBERT). It takes user input such as news title, content, subject, and date, and predicts whether the news is REAL or FAKE.

📝 Introduction
Fake news undermines public trust and spreads misinformation. Using NLP and Transformers, this project aims to accurately detect fake news by learning contextual cues from news data.

📊 Dataset
Source: Kaggle - Fake and Real News Dataset

Classes: Fake (0), Real (1)

Total Records: ~40,000 articles

Fields: Title, Text, Subject, Date

⚙️ Methodology
🔧 Preprocessing
HTML removal using BeautifulSoup

Special character and punctuation removal

Combined title, text, subject, and date into a single input string

🤖 Model
Fine-tuned DistilBERT using Hugging Face Transformers

Trained on labeled news articles

Tokenizer and model are saved and loaded during deployment

🧠 Architecture
text
Copy
Edit
[Title + Text + Subject + Date] → Clean → Tokenize → Transformer → Classification (Real/Fake)
🧪 Exploratory Data Analysis (EDA)
Distribution of labels: Balanced real/fake

Subject-wise article count

Word cloud of frequent terms in fake vs. real articles

📈 Results
Metric	Score
Accuracy	95%
F1-Score	94%
ROC-AUC	97%
Inference	< 1s

Confusion Matrix Example
(TP: 1800, FP: 90, TN: 1750, FN: 110)

💻 Code Structure
bash
Copy
Edit
.
├── app.py                       # Flask app for deployment
├── model/                       # Fine-tuned DistilBERT model
├── tokenizer/                   # Tokenizer folder
├── static/                      # CSS & styling
├── templates/
│   └── index.html               # HTML UI
├── requirements.txt             # Dependencies
└── README.md
🔍 App Flow
User inputs title, content, type, and date via web form

App cleans and processes the text

Text passed to Transformer model

Prediction is displayed as "REAL" or "FAKE"

🌐 Deployment
🔧 Local Setup
bash
Copy
Edit
git clone https://github.com/your-username/fake-news-transformer.git
cd fake-news-transformer
pip install -r requirements.txt
python app.py
🐳 Deployment Note (Render/Heroku)
Ensure the following:

gunicorn is added to requirements.txt

model/ and tokenizer/ directories are within your repo

Use relative paths (not absolute like E:\...) in AutoTokenizer.from_pretrained() and AutoModelForSequenceClassification.from_pretrained()

Replace:

python
Copy
Edit
tokenizer = AutoTokenizer.from_pretrained(r"E:\Fake_news_ditection\tokenizer")
model = AutoModelForSequenceClassification.from_pretrained(r"E:\Fake_news_ditection\model")
With:

python
Copy
Edit
tokenizer = AutoTokenizer.from_pretrained("tokenizer")
model = AutoModelForSequenceClassification.from_pretrained("model")
🛠️ Challenges & Solutions
Windows path error: Resolved by using relative paths and OS module

Model size & inference delay: Used DistilBERT for faster prediction

Deployment issues (e.g., gunicorn not found): Added gunicorn to requirements and checked PORT binding for Render

🚀 Future Enhancements
Real-time news scraping & prediction

Use multilingual transformers (e.g., XLM-RoBERTa)

Add explainability (e.g., attention heatmaps)

🧰 Technologies Used
Python

Flask

Hugging Face Transformers

BeautifulSoup

Torch

Qlik Sense, Power BI (for internal dashboards)
