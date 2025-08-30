from flask import Flask, request, jsonify
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer  # Import the Tokenizer class

# Initialisiere die Flask-Anwendung
app = Flask(__name__)

# Lade das vortrainierte KI-Modell
model = load_model('sentiment_model.h5')

# Maximaler Textlängenwert (für Padding)
max_length = 100

# Initialisiere den Tokenizer (muss vor der Nutzung des Analyze Sentiment durchgeführt werden)
tokenizer = Tokenizer()  # Assuming tokenizer is properly fitted elsewhere with training data

# Funktion für die Stimmungsanalyse
# Diese Funktion sollte den gegebenen Text tokenisieren, in die richtige Form bringen und das Modell nutzen, um die Stimmung vorherzusagen.
def analyze_sentiment(text):
    # Tokenize the input text and convert it to sequences
    text_sequence = tokenizer.texts_to_sequences([text])
    # Padding der Sequenz auf die maximale Länge
    text_sequence = pad_sequences(text_sequence, maxlen=max_length)
    # Vorhersage des Modells auf der Sequenz
    prediction = model.predict(text_sequence)
    # Dekodiere die Stimmung basierend auf der Vorhersage
    # Hier müssen die logischen Bedingungen zur Bestimmung der Stimmungswerte hinzugefügt werden
    sentiment = 'positive' if prediction[0][0] > 0.5 else 'negative'  # Using `prediction[0][0]` to access the value correctly
    return sentiment  # Rückgabe der ermittelten Stimmung

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    # Überprüfe, ob der erforderliche Text im JSON-Objekt vorhanden ist
    if 'text' not in data:
        return jsonify({'error': 'Kein Text angegeben.'}), 400

    text = data['text']
    sentiment = analyze_sentiment(text)
    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(debug=True)