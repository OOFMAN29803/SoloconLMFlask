from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import random

app = Flask(__name__)

# Load the model and tokenizer outside the route to save loading time
model = tf.keras.models.load_model('SoloconLM Beta 1.h5')
with open('tokenizerBeta1.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def sample_from_logits(logits, temperature=1.0):
    """ Apply temperature to logits and sample an index from the output probabilities. """
    scaled_logits = logits / temperature
    probabilities = tf.nn.softmax(scaled_logits).numpy()
    return np.random.choice(len(probabilities), p=probabilities)

def generate_text(model, tokenizer, seed_text, num_words=50):
    """ Generates text starting from a seed_text. """
    text_generated = seed_text
    for _ in range(num_words):
        encoded_text = tokenizer.texts_to_sequences([text_generated])
        pad_encoded = pad_sequences(encoded_text, maxlen=100, truncating='pre')
        pred_prob = model.predict(pad_encoded)
        pred_index = sample_from_logits(pred_prob[0], temperature=1.0)
        pred_word = tokenizer.index_word.get(pred_index, '[UNK]')
        text_generated += ' ' + pred_word
    return text_generated

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.get_json(force=True)
        seed_text = data.get('seed_text', '')
        num_words = data.get('num_words', 50)  # Default to 50 words if not specified
        if not seed_text.strip():
            return jsonify({'error': 'Seed text must not be empty.'}), 400
        generated_text = generate_text(model, tokenizer, seed_text, num_words)
        return jsonify({'generated_text': generated_text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
