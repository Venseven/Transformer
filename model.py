
import tensorflow as tf
# tf.debugging.set_log_device_placement(True)

# import imageio
import numpy as np
import pandas as pd
import random
import functools
from bert_serving.client import BertClient
import pathlib
import time
import os
import sys
from warnings import filterwarnings
filterwarnings("ignore")
from tqdm import tqdm
from tensorflow import keras
from tensorflow.keras import layers





class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(
            query, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(
            key, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(
            value, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(
            concat_attention
        )  # (batch_size, seq_len, embed_dim)
        return output


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.2):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class Generator_transformer(keras.Model):
    
    def __init__(self):
        super().__init__(name='generator')
        self.num_heads = 2
        self.ff_dim    = 32
        self.embed_dim = 1536
        self.vec_1 = tf.keras.layers.Conv1D(kernel_size=3, filters=200, padding='same',activation='tanh', strides=1)
        self.vec_2 = tf.keras.layers.MaxPool1D(1)

        self.transformer_block = TransformerBlock(self.embed_dim,self.num_heads, self.ff_dim)
        self.dense_1         = tf.keras.layers.Dense(units =1024)
        self.dense_2         = tf.keras.layers.Dense(units = 768)

    def call(self, input1,input2):
        x = self.transformer_block(tf.concat([input1,input2],axis = 2),training = True)
        vec = self.vec_1(input1)
        vec = self.vec_2(vec)
        vec1 = self.vec_1(input2)
        vec1 = self.vec_2(vec1)
        x = self.dense_1(tf.concat([x,vec,vec1],axis = 2))
        return self.dense_2(x)

        
class Generator_bilstm(keras.Model):
    
    def __init__(self):
        super().__init__(name='generator')
#         self.embed           = tf.keras.layers.Embedding()
        self.lstm            = tf.keras.layers.LSTM(units = 1024,return_sequences=True)
        self.lstm_back       = tf.keras.layers.LSTM(units = 1024,return_sequences=True,go_backwards = True)
        self.bi_1             = tf.keras.layers.Bidirectional(self.lstm,backward_layer = self.lstm_back)

        self.dense_1         = tf.keras.layers.Dense(units =1024)
        self.dense_2         = tf.keras.layers.Dense(units =1024)
        
        self.vec_1 = tf.keras.layers.Conv1D(kernel_size=3, filters=200, padding='same',activation='tanh', strides=1)
        self.vec_2 = tf.keras.layers.MaxPool1D(1)
        
        
        self.lstm1           = tf.keras.layers.LSTM(units = 1024,return_sequences=True)
        self.lstm_back2      = tf.keras.layers.LSTM(units = 1024,return_sequences=True,go_backwards = True)
        self.bi_2             = tf.keras.layers.Bidirectional(self.lstm1,backward_layer = self.lstm_back2)
        
        
        self.lstm3            = tf.keras.layers.LSTM(units = 900,return_sequences=True)
        self.lstm_back3      = tf.keras.layers.LSTM(units = 900, return_sequences=True,go_backwards = True)
        self.bi_3             = tf.keras.layers.Bidirectional(self.lstm3,backward_layer = self.lstm_back3)

#         self.lstm1           = tf.keras.layers.LSTM(units = 1024,return_sequences=True)

        self.dense_3         = tf.keras.layers.Dense(units =840)
        self.dense_4         = tf.keras.layers.Dense(units =840)

        
        self.lstm4            = tf.keras.layers.LSTM(units = 840,return_sequences=True)
        self.lstm_back4      = tf.keras.layers.LSTM(units = 840, return_sequences=True,go_backwards = True)
        self.bi_4             = tf.keras.layers.Bidirectional(self.lstm4,backward_layer = self.lstm_back4)

        self.dropout_layer_2 = tf.keras.layers.Dropout(0.2)
        self.dense_5         = tf.keras.layers.Dense(units =768)
        self.dropout_layer_3 = tf.keras.layers.Dropout(0.1)
        self.dense_6         = tf.keras.layers.Dense(units =768)
    
    def call(self, input1,input2):
        
        x = self.bi_1(tf.concat([input1,input2],axis = 2))
        x = self.bi_2(x)
        vec = self.vec_1(input1)
        vec = self.vec_2(vec)
        vec1 = self.vec_1(input2)
        vec1 = self.vec_2(vec1)
        x = tf.concat([x,vec,vec1],axis=2)
        x = self.bi_3(x)
        x = self.dense_4(x)
        
        
        return self.dense_6(x)
    
        
class Discriminator(keras.Model):
    def __init__(self):
        
        super().__init__(name = "discriminator")
        
        self.dense_1         = tf.keras.layers.Dense(units =200,activation = tf.nn.leaky_relu)
        
        self.dense_2         = tf.keras.layers.Dense(units =50,activation = tf.nn.leaky_relu)
        self.dense_3             = tf.keras.layers.Dense(units =100,activation = tf.nn.leaky_relu)
        self.dense_4         = tf.keras.layers.Dense(units =1)
        
    def call(self, input):
        
        x = self.dense_1(input)
        x = self.dense_2(x)
        
        return self.dense_4(x)
 
 