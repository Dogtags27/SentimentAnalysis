from flask import Flask, request, jsonify, render_template
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import json

app = Flask(__name__)

# Load BERT model and tokenizer
model = BertForSequenceClassification.from_pretrained("./Saved")

tokenizer = BertTokenizer.from_pretrained("./Saved")

# Function for sentiment prediction
def predict_sentiment(texts):
    if isinstance(texts, str):
        texts = [texts]
    print(texts)
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)

    # Map predicted class indices back to labels
    labels = {2: 'positive', 0: 'negative'}
    predicted_labels = [labels[pred.item()] for pred in predictions]

    return predicted_labels

def generate_word_cloud(text_list, output_filename,colormap):
    text = " ".join(text_list)
    wordcloud = WordCloud(width=800, height=400, background_color="black",colormap=colormap).generate(text)
    wordcloud.to_file(f'static/{output_filename}')

@app.route('/')
def home():
    return render_template("./home.html")

@app.route('/analyze', methods=['POST'])
def analyze():
    
    result = None
    
    # Check if form data was sent
    if 'single_input' in request.form and request.form['single_input']:
        # Process single string
        single_input = request.form['single_input']
        print(single_input)
        prediction = predict_sentiment(single_input)[0]
        return render_template('results.html', single_result=prediction)
    
    elif 'csv_file' in request.files and request.files['csv_file']:
        # Process uploaded CSV file
        csv_file = request.files['csv_file']
        if csv_file.filename.endswith('.csv'):
            csv_data = pd.read_csv(csv_file)  # Read CSV as DataFrame
            if 'Text' in csv_data.columns:
                text_list = csv_data['Text'].tolist()[110:130]
                predictions = predict_sentiment(text_list)
                
                # Count positive and negative sentiments
                sentiment_counts = {"positive": predictions.count("positive"), 
                                    "negative": predictions.count("negative")}

                # Generate word clouds
                if predictions.count("positive")>0:
                    generate_word_cloud([text_list[i] for i in range(len(predictions)) if predictions[i] == "positive"],
                                    "positive_wordcloud.png","cool")
                if predictions.count("negative")>0:
                    generate_word_cloud([text_list[i] for i in range(len(predictions)) if predictions[i] == "negative"],
                                    "negative_wordcloud.png","hot")

                return render_template('results.html', pie_data=sentiment_counts)
                
            else:
                return "Error: CSV file must have a 'Text' column.", 400
        else:
            return "Error: Please upload a valid CSV file.", 400
    
    else:
        return "Error: No valid input provided.", 400

if __name__ == "__main__":
    app.run(debug=True)
