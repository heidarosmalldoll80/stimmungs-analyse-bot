from flask import Flask, request, jsonify
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

# Initialisiere die Flask-Anwendung
app = Flask(__name__)

# Lade das vortrainierte KI-Modell
model = load_model('sentiment_model.h5')

# Maximaler Textlängenwert (für Padding)
max_length = 100

# Funktion für die Stimmungsanalyse
def analyze_sentiment(text):
    # Hier sollte der Text in geeignete Token umgewandelt werden
    # Placeholder für Tokenisierung und Sequenzierung
    # Beispiel: text_sequence = tokenizer.texts_to_sequences([text])
    # text_sequence = pad_sequences(text_sequence, maxlen=max_length)
    # prediction = model.predict(text_sequence)
    # return decoded sentiment based on prediction
    return 'neutral'  # Temporär, für die Demo

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    if 'text' not in data:
        return jsonify({'error': 'Kein Text angegeben.'}), 400

    text = data['text']
    sentiment = analyze_sentiment(text)
    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(debug=True)
