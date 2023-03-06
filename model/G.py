from __future__ import absolute_import, division, print_function, unicode_literals
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

class Encoder(tf.keras.Model):

  def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
    super(Encoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = enc_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

  def call(self, x):
    x = self.embedding(x)
    output, state = self.gru(x)
    return output, state



class BahdanauAttention(tf.keras.layers.Layer):

  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):
    hidden_with_time_axis = tf.expand_dims(query, 1)
    score = self.V(tf.nn.tanh(
        self.W1(values) + self.W2(hidden_with_time_axis)))
    attention_weights = tf.nn.softmax(score, axis=1)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)
    return context_vector, attention_weights



class Decoder(tf.keras.Model):

  def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
    super(Decoder, self).__init__()
    self.batch_sz = batch_sz
    self.dec_units = dec_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc = tf.keras.layers.Dense(vocab_size)
    self.attention = BahdanauAttention(self.dec_units)

  def call(self, x, hidden, enc_output):
    context_vector, attention_weights = self.attention(hidden, enc_output)
    x = self.embedding(x)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
    output, state = self.gru(x)
    output = tf.reshape(output, (-1, output.shape[2]))
    x = self.fc(output)
    return x, state, context_vector


class Seq2Seq(tf.keras.Model):

  def __init__(self, encoder, decoder, targ_lang):
    super(Seq2Seq, self).__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.vocab_tar_size = targ_lang
  def call(self, inp , targ, teacher_forcing_ratio=0.8):
      enc_output, enc_hidden = self.encoder(inp)
      dec_hidden = enc_hidden
      dec_input = tf.expand_dims(targ[:,0], 1)
      total_seq = tf.one_hot(dec_input,depth = self.vocab_tar_size)
      for t in range(1, targ.shape[1]):
          predictions, dec_hidden, context = self.decoder(dec_input, dec_hidden, enc_output)
          is_teacher = tf.random.uniform([], minval=0, maxval=1) < teacher_forcing_ratio
          output = targ[:, t] if is_teacher else tf.argmax(predictions,-1,output_type='int32')

          dec_input = tf.expand_dims(output,1) # next input
          total_seq = tf.concat([total_seq,tf.expand_dims(predictions, 1)],axis=1)

      return total_seq
