import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import random
import json
import pickle

# To Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# To Load data
file_path = '/content/drive/MyDrive/intents.json'
with open(file_path, 'r') as file:
    data = json.load(file)

# To Process or load pre-processed data
try:
    with open("/content/drive/MyDrive/data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words, labels, docs_x, docs_y = [], [], [], []
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = sorted(set([stemmer.stem(w.lower()) for w in words if w != "?"]))
    labels = sorted(labels)

    out_empty = [0] * len(labels)
    training, output = [], []
    for x, doc in enumerate(docs_x):
        bag = [1 if stemmer.stem(w.lower()) in doc else 0 for w in words]
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training, output = np.array(training), np.array(output)
    with open("/content/drive/MyDrive/data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

# To Define model
model = Sequential([
    Dense(8, activation='relu', input_shape=(len(training[0]),)),
    Dense(8, activation='relu'),
    Dense(len(output[0]), activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# To Load or train model
try:
    model.load_weights("/content/drive/MyDrive/model.h5")
except:
    model.fit(training, output, epochs=1000, batch_size=8, verbose=1)
    model.save_weights("/content/drive/MyDrive/model.h5")

# To Utility function for creating a bag of words
def bag_of_words(s, words):
    bag = [0] * len(words)
    s_words = [stemmer.stem(word.lower()) for word in nltk.word_tokenize(s)]
    for w in s_words:
        if w in words:
            bag[words.index(w)] = 1
    return np.array([bag])

# To Chat function
def chat():
    print("Start talking with the bot (type quit to stop)!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict(bag_of_words(inp, words))
        results_index = np.argmax(results)
        tag = labels[results_index]

        if results[0][results_index] > 0.7:  # Adjusting for the correct dimension
            responses = next((tg['responses'] for tg in data['intents'] if tg['tag'] == tag), ["Not sure what you mean"])
            print(random.choice(responses))
        else:
            print("I'm not sure what you want.")

# To Start the chat
chat()
