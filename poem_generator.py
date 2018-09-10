import tensorflow as tf  # version 1.9 or above
tf.enable_eager_execution()  # Execution of code as it runs in the notebook. Normally, TensorFlow looks up the whole code before execution for efficiency.

import numpy as np
import re
import random
import unidecode
import time


path_to_file = 'poem_corpus.txt'
text = unidecode.unidecode(open(path_to_file).read())

unique = sorted(set(text))  # unique contains all the unique characters in the corpus

char2idx = {u:i for i, u in enumerate(unique)}  # maps characters to indexes
idx2char = {i:u for i, u in enumerate(unique)}  # maps indexes to characters

# Hyperparameters
max_length = 100  # Maximum length sentence we want per input in the network
vocab_size = len(unique)
embedding_dim = 256  # number of 'meaningful' features to learn. Ex: ['queen', 'king', 'man', 'woman'] has a least 2 embedding dimension: royalty and gender.
units = 1024  # In keras: number of output of a sequence. In short it rem
BATCH_SIZE = 64
BUFFER_SIZE = 10000


input_text = []
target_text = []

for f in range(0, len(text) - max_length, max_length):
	inps = text[f : f + max_length]
	targ = text[f + 1 : f + 1 + max_length]
	input_text.append([char2idx[i] for i in inps])
	target_text.append([char2idx[t] for t in targ])
    
dataset = tf.data.Dataset.from_tensor_slices((input_text, target_text)).shuffle(BUFFER_SIZE)
dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(BATCH_SIZE))


class Model(tf.keras.Model):
	def __init__(self, vocab_size, embedding_dim, units, batch_size):
		super(Model, self).__init__()
		self.units = units
		self.batch_sz = batch_size
		self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
		if tf.test.is_gpu_available():
			self.gru = tf.keras.layers.CuDNNGRU(self.units, 
												return_sequences=True, 
												return_state=True, 
												recurrent_initializer='glorot_uniform')
		else:
			self.gru = tf.keras.layers.GRU(self.units, 
											return_sequences=True, 
											return_state=True, 
											recurrent_activation='sigmoid', 
											recurrent_initializer='glorot_uniform')
		self.fc = tf.keras.layers.Dense(vocab_size)
		
	def call(self, x, hidden):
		x = self.embedding(x)
		output, states = self.gru(x, initial_state=hidden)
		output = tf.reshape(output, (-1, output.shape[2]))
		x = self.fc(output)
		return x, states

model = Model(vocab_size, embedding_dim, units, BATCH_SIZE)

optimizer = tf.train.AdamOptimizer()

# using sparse_softmax_cross_entropy so that we don't have to create one-hot vectors
def loss_function(real, preds):
	return tf.losses.sparse_softmax_cross_entropy(labels=real, logits=preds)


# Training step

EPOCHS = 30

for epoch in range(EPOCHS):
	start = time.time()
	hidden = model.reset_states()
	for (batch, (inp, target)) in enumerate(dataset):
		with tf.GradientTape() as tape:
			predictions, hidden = model(inp, hidden)
			target = tf.reshape(target, (-1,))
			loss = loss_function(target, predictions)
		
		grads = tape.gradient(loss, model.variables)
		optimizer.apply_gradients(zip(grads, model.variables), global_step=tf.train.get_or_create_global_step())
		
		if batch % 100 == 0:
			print ('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, loss))
			
	print ('Epoch {} Loss {:.4f}'.format(epoch + 1, loss))
	print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
    
	
# Evaluation step(generating text using the model learned)

num_generate = 1000  # number of characters to generate
start_string = 'The child'  # beginning of the generated text. TODO: try start_string = ' '

input_eval = [char2idx[s] for s in start_string]  # converts start_string to numbers the model understands
input_eval = tf.expand_dims(input_eval, 0)  # 

text_generated = ''

temperature = 0.97  # the greater, the closer to an observation in the corpus

hidden = [tf.zeros((1, units))]
for i in range(num_generate):
	predictions, hidden = model(input_eval, hidden)  # predictions holds the probabily for each character to be most adequate continuation
	
	predictions = predictions / temperature  # alters characters' probabilities to be picked (but keeps the order)
	predicted_id = tf.multinomial(tf.exp(predictions), num_samples=1)[0][0].numpy()  # picks the next character for the generated text
	
	input_eval = tf.expand_dims([predicted_id], 0)
	text_generated += idx2char[predicted_id]

print (start_string + text_generated)
