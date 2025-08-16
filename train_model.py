import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Input
import random
import json
import pickle
import os

# Create a stemmer for word normalization
stemmer = LancasterStemmer()

# Download NLTK's tokenizer model (only needed once)
#nltk.download('punkt')

# ---------------------------
# LOAD INTENTS FROM JSON FILE
# ---------------------------
with open("intents.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# ---------------------------
# LOAD OR CREATE TRAINING DATA
# ---------------------------
DATA_PICKLE = "data.pickle"
try:
    # Try loading preprocessed data to save time
    with open(DATA_PICKLE, "rb") as f:
        words, labels, training, output = pickle.load(f)
except FileNotFoundError:
    # Variables to store vocabulary, labels, and training samples
    words = []
    labels = []
    docs_x = []  # tokenized patterns
    docs_y = []  # corresponding tags

    # Loop through each intent
    for intent in data["intents"]:
        for pattern in intent.get("patterns", []):
            # Tokenize the pattern into words
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        # Add label if not already in the list
        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    # Stem words (reduce to base form) and remove duplicates
    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))
    labels = sorted(labels)

    # Create bag-of-words and one-hot label arrays
    training = []
    output = []
    out_empty = [0 for _ in range(len(labels))]  # base vector for labels

    for x, doc in enumerate(docs_x):
        bag = []

        # Stem each word in the pattern
        wrds = [stemmer.stem(w.lower()) for w in doc]

        # Create bag-of-words: 1 if word exists in pattern, else 0
        for w in words:
            bag.append(1 if w in wrds else 0)

        # One-hot encode the label
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    # Convert to NumPy arrays for TensorFlow
    training = np.array(training, dtype=np.float32)
    output = np.array(output, dtype=np.float32)

    # Save processed data for future runs
    with open(DATA_PICKLE, "wb") as f:
        pickle.dump((words, labels, training, output), f)
    print("✅ Preprocessed data saved.")

# ---------------------------
# DEFINE MODEL FUNCTION
# ---------------------------
def build_model(input_size, output_size):
    model = Sequential([
        Input(shape=(input_size,)),              # Input layer
        Dense(8, activation="relu"),             # First hidden layer
        Dense(8, activation="relu"),             # Second hidden layer
        Dense(output_size, activation="softmax") # Output layer
    ])
    # Compile model with Adam optimizer & categorical crossentropy loss
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

# Train and save the model
MODEL_FILE = "model.keras"
print("--- Starting Model Training ---")
model = build_model(len(training[0]), len(output[0]))
model.fit(training, output, epochs=500, batch_size=8, verbose=1)
model.save(MODEL_FILE)
print("Model trained and saved to", MODEL_FILE)