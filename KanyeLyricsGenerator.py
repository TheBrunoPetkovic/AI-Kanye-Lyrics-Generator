import random
import numpy as np
import tensorflow as tf
from keras.api.models import load_model

filename = "Kanye West Lyrics.txt"
with open(filename, "r", encoding="utf-8-sig") as f:
   data = f.read().lower()

characters = sorted(set(data))

char_to_index = dict((c, i) for i, c in enumerate(characters))
index_to_char = dict((i, c) for i, c in enumerate(characters))

# Hyperparameters and declarations
SEQ_LENGTH = 50
STEP_SIZE = 1

sentences = []
next_char = []

trained_model = load_model("Generator.keras")

def sample(preds, temperature):
   preds = np.asarray(preds).astype('float64')
   preds = np.log(preds) / temperature
   exp_preds = np.exp(preds)
   preds = exp_preds / np.sum(exp_preds)
   probas = np.random.multinomial(1, preds, 1)
   return np.argmax(probas)

def generate_text(length, temperature):
   start_index = random.randint(0, len(data) - SEQ_LENGTH - 1)
   generated = ""
   sentence = data[start_index: start_index + SEQ_LENGTH]
   generated += sentence
   for i in range(length):
      x_predictions = np.zeros((1, SEQ_LENGTH, len(characters)))
      for t, char in enumerate(sentence):
         x_predictions[0, t, char_to_index[char]] = 1

      predictions = trained_model.predict(x_predictions, verbose=0)[0]
      next_index = sample(predictions, temperature)
      next_character = index_to_char[next_index]

      generated += next_character
      sentence = sentence[1:] + next_character
   return generated

print("-----------0.1--------")
print(generate_text(500, 0.1))
print("-----------0.2--------")
print(generate_text(500, 0.2))
print("-----------0.3--------")
print(generate_text(500, 0.3))
print("-----------0.4--------")
print(generate_text(500, 0.4))
print("-----------0.5--------")
print(generate_text(500, 0.5))
print("-----------0.6--------")
print(generate_text(500, 0.6))
print("-----------0.7--------")
print(generate_text(500, 0.7))
print("-----------0.8--------")
print(generate_text(500, 0.8))
print("-----------0.9--------")
print(generate_text(500, 0.9))
print("-----------1--------")
print(generate_text(500, 1.0))
