import re
import logging
import torch
import pandas as pd
from flask import Flask, request, jsonify
from flasgger import Swagger
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import spacy

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TextPreprocessor:
    def __init__(self):
        """Initializes text preprocessing components."""
        self.stopwords = set(word.lower() for word in stopwords.words('english')).union({
            'i', 'the', 'these', 'there', 'are', 'this', 'that', 'we', 'you', 'it', 'they', 'he', 'she', 'them', 'is', 'am', 'was', 'were', 'been', 'being'
        })
        self.lemmatizer = WordNetLemmatizer()
    
    def clean_text(self, text):
        """Applies various text preprocessing steps."""
        text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
        text = text.encode('ascii', 'ignore').decode('utf-8')  # Remove accented chars
        text = re.sub(r'[̀-ͯ]', '', text)  # Replace diacritics
        contractions = {"can't": "cannot", "won't": "will not", "it's": "it is"}  # Expand contractions
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        text = re.sub(r'http\S+', '', text)  # Remove URLs
        text = re.sub(r'[:;=X][oO\-]?[D\)\(P]', '', text)  # Remove emojis
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
        text = re.sub(r'\d+', '', text)  # Remove numbers
        text = " ".join([word.lower() for word in text.split() if word.lower() not in self.stopwords])
        text = " ".join([self.lemmatizer.lemmatize(word) for word in text.split()])
        return re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace

class SentimentAnalyzer:
    def __init__(self, model_name='nlptown/bert-base-multilingual-uncased-sentiment'):
        """Loads the sentiment analysis model and tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    def analyze_sentiment(self, text):
        """Encodes text and predicts sentiment score."""
        tokens = self.tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=512)
        result = self.model(tokens)
        return int(torch.argmax(result.logits)) + 1  # Convert to 1-5 scale

class EntityExtractor:
    def __init__(self):
        """Loads NLP models for entity recognition and dependency parsing."""
        self.ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple")
        self.nlp = spacy.load("en_core_web_md")
    
    def extract_entities(self, text):
        """Extracts key subjects and objects using dependency parsing and NER."""
        doc = self.nlp(text)
        subjects, objects = [], []
        
        for token in doc:
            if "subj" in token.dep_:
                subjects.append(token.text)
            if "obj" in token.dep_ or token.pos_ == "NOUN":
                objects.append(token.text)
        
        entities = self.ner_pipeline(text)
        for entity in entities:
            if entity["entity_group"] in ["PER", "ORG","LOC","PROD"]:
                subjects.append(entity["word"])
            elif entity["entity_group"] in ["LOC", "MISC"]:
                objects.append(entity["word"])
        
        if not subjects:
            subjects = [token.text for token in doc if token.pos_ == "NOUN"][:1]  
        if not objects:
            objects = [token.text for token in doc if token.pos_ == "NOUN"][:1]  

        return {"subject": " ".join(subjects) if subjects else "Unknown", "object": " ".join(objects) if objects else "Unknown"}

# Initialize components
logger.info("Initializing components...")
preprocessor = TextPreprocessor()
sentiment_analyzer = SentimentAnalyzer()
entity_extractor = EntityExtractor()

# Flask App
app = Flask(__name__)
Swagger(app)

@app.route('/analyze', methods=['POST'])
def analyze():
    """API endpoint to analyze sentiment and extract entities.
    ---
    parameters:
      - name: text
        in: body
        required: true
        schema:
          type: object
          properties:
            text:
              type: string
              example: "The app’s search feature is accurate and fast."
    responses:
      200:
        description: Sentiment analysis result
        schema:
          type: object
          properties:
            sentiment:
              type: integer
              example: 5
            subject:
              type: string
              example: "app"
            object:
              type: string
              example: "fantastic"
      400:
        description: Invalid input
      500:
        description: Internal server error
    """
    try:
        data = request.get_json()
        user_text = data.get('text', '')
        if not user_text:
            return jsonify({"error": "No text provided"}), 400
        
        cleaned_text = preprocessor.clean_text(user_text)
        sentiment = sentiment_analyzer.analyze_sentiment(cleaned_text)
        entities = entity_extractor.extract_entities(cleaned_text)
        
        return jsonify({
            "sentiment": sentiment,
            "subject": entities["subject"],
            "object": entities["object"]
        })
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=6001)

