import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re
import string
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from flask import Flask, render_template, request
import os

tokenizer_path = os.path.join("tokenizer")
model_path = os.path.join("model")

tokenizer = AutoTokenizer.from_pretrained(r"E:\Fake_news_ditection\tokenizer")
model = AutoModelForSequenceClassification.from_pretrained(r"E:\Fake_news_ditection\model")

def remove_html(text):
    """Removes HTML tags from text."""
    soup = BeautifulSoup(text, 'html.parser')
    return soup.get_text()

def remove_special_characters(text):
    """Removes special characters (non-alphanumeric and non-whitespace)."""
    pattern = r'[^a-zA-Z0-9\s]'
    return re.sub(pattern, '', text)

def remove_punctuation(text):
    """Removes punctuation using string.punctuation."""
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

def clean_text(text):
    """Applies all cleaning functions to the text."""
    if isinstance(text, str):  # Ensure the input is a string
        text = remove_html(text)
        text = remove_special_characters(text)
        text = remove_punctuation(text)
        return text
    return text 

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
    return predicted_class

app = Flask(__name__)

@app.route("/", methods = ["POST" , "GET"])
def homepage():
    result = None
    label = None

    if request.method == "POST":
        title = request.form.get("news_title", "")
        content = request.form.get("news_content", "")
        subject = request.form.get("news_type", "")
        date = request.form.get("news_date", "")
        print(type(subject))
        clean_title = clean_text(title)
        clean_content = clean_text(content)
        combined_text = clean_title + " " + clean_content + " " + str(subject) + " " + str(date)
        print(combined_text)
        label = predict(combined_text)
        result = "REAL" if label == 1 else "FAKE"

    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
