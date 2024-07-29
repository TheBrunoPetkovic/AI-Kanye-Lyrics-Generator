import sys
import io
import numpy as np
from keras.api.models import Sequential
from keras.api.optimizers import RMSprop
from keras.api.layers import Activation, Dense, LSTM

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

filename = "Kanye West Lyrics.txt"
with open(filename, "r", encoding="utf-8-sig") as f:
   data = f.read().lower()

characters = sorted(set(data))

char_to_index = dict((c, i) for i, c in enumerate(characters))
index_to_char = dict((i, c) for i, c in enumerate(characters))

# Hyperparameters and declarations
SEQ_LENGTH = 50
STEP_SIZE = 1
LEARNING_RATE = 0.01
BATCH_SIZE = 256
NUM_EPOCHS = 4

sentences = []
next_char = []

for i in range(0, len(data) - SEQ_LENGTH, STEP_SIZE):
   sentences.append(data[i: i + SEQ_LENGTH])
   next_char.append(data[i + SEQ_LENGTH])
   
x = np.zeros((len(sentences), SEQ_LENGTH, len(characters)), dtype=bool)
y = np.zeros((len(sentences), len(characters)), dtype=bool)

for i, line in enumerate(sentences):
   for t, char in enumerate(line):
      x[i, t, char_to_index[char]] = 1
   y[i, char_to_index[next_char[i]]] = 1

model = Sequential()
model.add(LSTM(128, input_shape=(SEQ_LENGTH, len(characters))))
model.add(Dense(len(characters)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=LEARNING_RATE))

model.fit(x, y, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS)

model.save("Generator.keras")