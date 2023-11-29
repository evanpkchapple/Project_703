import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

### IMPORTS ###

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import LSTM, Embedding, Dense, Input
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import os
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

### ATTENTION LAYER ###

class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def call(self, query, values):
        query_with_time_axis = tf.expand_dims(query, 1)

        score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))

        attention_weights = tf.nn.softmax(score, axis=1)

        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

### ENCODER LAYER ###

class Encoder(Model):
    def __init__(self, num_chars, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units

        self.embedding = Embedding(num_chars, embedding_dim)

        self.lstm = LSTM(self.enc_units, return_sequences=True, return_state=True)

    def call(self, x, hidden):
        x = self.embedding(x)
        output, hidden_state, cell_state = self.lstm(x, initial_state=hidden)
        return output, [hidden_state, cell_state]

    def initialize_hidden_state(self):
        return [tf.zeros((self.batch_sz, self.enc_units)) for _ in range(2)]

### DECODER LAYER ###

class Decoder(Model):
    def __init__(self, num_chars, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(num_chars, embedding_dim)
        self.lstm = tf.keras.layers.LSTM(self.dec_units, return_sequences=True, return_state=True)
        self.fc = tf.keras.layers.Dense(num_chars)

        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden_states, enc_output):
        context_vector, attention_weights = self.attention(hidden_states[0], enc_output)

        x = self.embedding(x)

        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        output, hidden_state, cell_state = self.lstm(x)

        output = tf.reshape(output, (-1, output.shape[2]))

        x = self.fc(output)

        return x, [hidden_state, cell_state], attention_weights

### LOADING DATA ###

data_path = 'data/ipa_out.txt'

f = open(data_path, 'r')
source_words = []
target_words = []

for line in f:
    word = line.strip().split('$')
    source_words.append('$' + word[0] + '#')
    target_words.append('$' + word[1] + '#')

print('Length of training data: ', len(source_words))

all_chars = set(''.join(source_words) + ''.join(target_words) + ' ')
char_to_int = {c: i for i, c in enumerate(all_chars)}
int_to_char = {i: c for c, i in char_to_int.items()}

max_len_source = max(len(word) for word in source_words)

max_len_target = max(len(word) for word in target_words)

max_len = max(max_len_source, max_len_target)

def word_to_int_seq(word):
    return [char_to_int[char] for char in word]

X = [word_to_int_seq(word) for word in source_words]
Y = [word_to_int_seq(word) for word in target_words]

X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=max_len, padding='post', value=char_to_int[' '])
Y = tf.keras.preprocessing.sequence.pad_sequences(Y, maxlen=max_len, padding='post', value=char_to_int[' '])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

### HYPER PARAMETER DEFINITION ###

BUFFER_SIZE = len(X_train)
BATCH_SIZE = 64
steps_per_epoch = len(X_train)//BATCH_SIZE
num_chars = len(all_chars)
embedding_dim = 256
units = 1024
EPOCHS = 15

### CREATE DATASET ###

dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

example_input_batch, example_target_batch = next(iter(dataset))
print(example_input_batch.shape, example_target_batch.shape)

### TEST ENCODER ###

encoder = Encoder(num_chars, embedding_dim, units, BATCH_SIZE)

sample_hidden = encoder.initialize_hidden_state()
sample_output, sample_states = encoder(example_input_batch, sample_hidden)

print ('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
print ('Encoder Hidden state shape: (batch size, units) {}'.format(sample_states[0].shape))
print ('Encoder Cell state shape: (batch size, units) {}'.format(sample_states[1].shape))

### TEST ATTENTION ###

attention_layer = BahdanauAttention(10)
attention_result, attention_weights = attention_layer(sample_states[0], sample_output)

print("Attention result shape: (batch size, units) {}".format(attention_result.shape))
print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))

### TEST DECODER ###

decoder = Decoder(num_chars, embedding_dim, units, BATCH_SIZE)

sample_decoder_output, _, _ = decoder(tf.random.uniform((BATCH_SIZE, 1)),
                                      sample_states, sample_output)

print ('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))

### LOSS & OPTIMIZER ###

optimizer = tf.keras.optimizers.Adam()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):

  mask = tf.math.logical_not(tf.math.equal(real, char_to_int[' ']))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)

### Training Function ###

@tf.function
def train_step(input, target, enc_hidden):
    loss = 0

    with tf.GradientTape() as tape:
        enc_output, enc_states = encoder(input, enc_hidden)
        
        dec_states = enc_states

        dec_input = tf.expand_dims([char_to_int['$']] * BATCH_SIZE, 1)

        for t in range(1, target.shape[1]):
            predictions, dec_states, _ = decoder(dec_input, dec_states, enc_output)
            
            loss += loss_function(target[:, t], predictions)

            dec_input = tf.expand_dims(target[:, t], 1)

        batch_loss = (loss / int(target.shape[1]))

        variables = encoder.trainable_variables + decoder.trainable_variables

        gradients = tape.gradient(loss, variables)

        optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss

### Training LOOP ###

for epoch in range(EPOCHS):
    start = time.time()

    enc_states = encoder.initialize_hidden_state()
    total_loss = 0

    for (batch, (input, target)) in enumerate(dataset.take(steps_per_epoch)):
        batch_loss = train_step(input, target, enc_states)

        total_loss += batch_loss

        if batch % 20 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch+1, batch, batch_loss.numpy()))
    if (epoch+1) % 2 == 0:
        checkpoint.save(file_prefix = checkpoint_prefix)

    print('Epoch {} Loss {:.4f}'.format(epoch+1, total_loss/steps_per_epoch))

    print('Time taken for 1 epoch {} sec\n'.format(time.time()-start))

### EVALUATE ###

def evaluate(word):
    attention_plot = np.zeros((max_len, max_len))

    word = "$" + word + "#"

    print(word)

    input_seq = word_to_int_seq(word)

    print(input_seq)

    input = tf.keras.preprocessing.sequence.pad_sequences([input_seq], maxlen=max_len, padding='post', value=char_to_int[' '])

    inputs = tf.convert_to_tensor(input)

    result = ''

    hidden_state = [tf.zeros((1, units)) for _ in range(2)]

    enc_out, enc_hidden = encoder(inputs, hidden_state)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([char_to_int['$']], 0)

    for t in range(max_len):
        predictions, dec_states, attention_weights = decoder(dec_input, dec_hidden, enc_out)

        attention_weights = tf.reshape(attention_weights, (-1, ))
        attention_plot[t] = attention_weights.numpy()

        predicted_id = tf.argmax(predictions[0]).numpy() #doesn't do much

        result += int_to_char[predicted_id]

        if int_to_char[predicted_id] == "#":
            return result, word, attention_plot
            
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, word, attention_plot

def plot_attention(attention, word, predicted_word):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')

    fontdict = {'fontsize': 14}

    ax.set_xticklabels([''] + word, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_word, fontdict=fontdict)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()

def translate(word):
    result, word, attention_plot = evaluate(word)

    print('Input: %s' % (word))
    print('Predicted translation: {}'.format(result))

    attention_plot = attention_plot[:len([*result]), :len([*word])]
    plot_attention(attention_plot, [*word], [*result])

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

translate("kulinaɾjas")