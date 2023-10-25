import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from levenstein import levenstein_distance

f = open('data/ipa_out.txt', 'r')
source_words = []
target_words = []

for line in f:
    word = line.strip().split('$')
    print(word)
    source_words.append(word[0])
    target_words.append(word[1])

print('Length of training data: ', len(source_words))

all_chars = set(''.join(source_words) + ''.join(target_words) + ' ')
char_to_int = {c: i for i, c in enumerate(all_chars)}

max_len = max(max(len(word) for word in source_words), max(len(word) for word in target_words))

def word_to_int_seq(word):
    return [char_to_int[char] for char in word]

X = [word_to_int_seq(word) for word in source_words]
Y = [word_to_int_seq(word) for word in target_words]

X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=max_len, padding='post', value=char_to_int[' '])
Y = tf.keras.preprocessing.sequence.pad_sequences(Y, maxlen=max_len, padding='post', value=char_to_int[' '])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

input_shape = (max_len, )
model = tf.keras.Sequential([
tf.keras.layers.InputLayer(input_shape=input_shape),
tf.keras.layers.Embedding(len(all_chars), 128, input_length=max_len),
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(512, activation='relu'),
tf.keras.layers.Dense(max_len * len(all_chars), activation='softmax'),
tf.keras.layers.Reshape((max_len, len(all_chars)))
])

loss_fn = tf.losses.CategoricalCrossentropy()
optimizer = tf.optimizers.Adam()

model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
model.fit(X_train, tf.one_hot(Y_train, depth=len(all_chars)), epochs=10, batch_size=32, validation_split=0.1)

predictions = model.predict(X_test)
predicted_sequences = np.argmax(predictions, axis=-1)

int_to_char = {i: c for c, i in char_to_int.items()}
predicted_words = [''.join([int_to_char[int_val] for int_val in seq]) for seq in predicted_sequences]
#print(predicted_words)

X_test_words = [''.join([int_to_char[int_val] for int_val in seq]) for seq in X_test]
Y_test_words = [''.join([int_to_char[int_val] for int_val in seq]) for seq in Y_test]

for i in range(len(X_test)):
    print('Origin: ',X_test_words[i].strip(), '\t -> ', predicted_words[i].strip(), '\t [', Y_test_words[i].strip(), ']\nLevenstein Distance: ', levenstein_distance(X_test_words[i], predicted_words[i]).distance)
