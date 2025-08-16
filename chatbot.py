import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import random
import json
import pickle
import os

# Create a stemmer for word normalization
stemmer = LancasterStemmer()

# ---------------------------
# LOAD INTENTS AND DATA
# ---------------------------
with open("intents.json", "r", encoding="utf-8") as file:
    data = json.load(file)

DATA_PICKLE = "data.pickle"
try:
    with open(DATA_PICKLE, "rb") as f:
        words, labels, training, output = pickle.load(f)
    print("✅ Preprocessed data loaded successfully.")
except FileNotFoundError:
    print("⚠️ Error: Preprocessed data file 'data.pickle' not found.")
    print("Please run 'train_model.py' first to generate it.")
    exit()

# Load the trained model
MODEL_FILE = "model.keras"
try:
    model = load_model(MODEL_FILE)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"⚠️ Error loading model: {e}")
    print("ℹ️ Please run the training script 'train_model.py' first.")
    exit()

# ---------------------------
# HELPER FUNCTIONS
# ---------------------------
def bag_of_words(s, words):
    """
    Convert a user sentence into a bag-of-words array based on known vocabulary.
    """
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return np.array(bag, dtype=np.float32)

# ---------------------------
# CHATBOT RESPONSE FUNCTION
# ---------------------------

def get_chatbot_response(user_input):
    """
    Processes a single user input and returns a chatbot response.
    """
    bow = bag_of_words(user_input, words)
    results = model.predict(np.array([bow]), verbose=0)[0]
    results_index = np.argmax(results)
    tag = labels[results_index]
    
    # Check confidence threshold
    if results[results_index] > 0.7:
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg.get('responses', [])
                break
            # Debugging info
            #print(f"Confidence scores: {results}")
            #print(f"Predicted intent: {tag} with confidence {results[results_index]:.2f}")
        if responses:
            return random.choice(responses)
    return "My apologies, could you maybe rephrase that?"
