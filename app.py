from flask import Flask, request, render_template, jsonify
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os

app = Flask(__name__)

# Set the NLTK data path to the local directory
nltk.data.path.append(os.path.join(os.path.dirname(__file__), 'nltk_data'))

# Initialize the SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form['text']
    sentiment = sia.polarity_scores(text)

    if sentiment['compound'] > 0.05:
        sentiment_type = 'Positive'
    elif sentiment['compound'] < -0.05:
        sentiment_type = 'Negative'
    else:
        sentiment_type = 'Neutral'

    response = {
        'text': text,
        'sentiment_type': sentiment_type,
        'compound': sentiment['compound'],
        'positive': sentiment['pos'],
        'negative': sentiment['neg'],
        'neutral': sentiment['neu']
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
