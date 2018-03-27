
# coding: utf-8

# In[1]:


import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from keras.backend.tensorflow_backend import set_session
set_session(session)

from keras.callbacks import LambdaCallback,ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM,CuDNNLSTM, Dropout
from keras.optimizers import RMSprop,Adam
from keras.utils.data_utils import get_file
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger

import math

import numpy as np
import random
import sys
import io
import dill as pickle
import os

import nltk
import codecs


# In[2]:


class CharLSTM:
    
    def prepare_training_data(self,text):
        #text = self.read_file(train_path)
        chars = sorted(list(set(text)))
        print('total chars:', len(chars))
        char_indices = dict((c, i) for i, c in enumerate(chars))
        indices_char = dict((i, c) for i, c in enumerate(chars))
        maxlen = self.bptt
        step = 3
        sentences = []
        next_chars = []
        for i in range(0, len(text) - maxlen, step):
            sentences.append(text[i: i + maxlen])
            next_chars.append(text[i + maxlen])
        print('number of sequences:', len(sentences))

        print('Vectorization...')
        x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
        y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
        for i, sentence in enumerate(sentences):
            for t, char in enumerate(sentence):
                x[i, t, char_indices[char]] = 1
            y[i, char_indices[next_chars[i]]] = 1
            
        self.chars = chars
        self.char_indices = char_indices
        self.indices_char = indices_char
        self.text = text
        data_utils = DataUtils()
        vocab = {'chars':self.chars , 'char_indices':self.char_indices, 'text':self.text, 'bptt':self.bptt,
                'indices_char':self.indices_char}
        data_utils.pickler(self.vocab_save_path,"vocab.pkl",vocab)
        
        return x,y
    
    def build_model(self,learning_rate):
        model = Sequential()
        model.add(CuDNNLSTM(self.dim_hidden,return_sequences=True,input_shape=(self.bptt, len(self.chars))))
        model.add(Dropout(0.5))
        model.add(CuDNNLSTM(self.dim_hidden,return_sequences=True,input_shape=(self.bptt, len(self.chars))))
        model.add(Dropout(0.5))
        model.add(CuDNNLSTM(self.dim_hidden,input_shape=(self.bptt, len(self.chars))))
        model.add(Dropout(0.5))
        model.add(Dense(len(self.chars)))
        model.add(Activation('softmax'))
        #optimizer = RMSprop(lr=learning_rate)
        optimizer = Adam(lr=learning_rate,clipvalue=0.25)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer)
        return model
    
    def sample(self,preds, temperature=1.0):
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)
    
    def save_model(self,save_path):
        self.model.save(save_path)
    
    def save_model_weights(self,save_path):
        self.model.save_weights(save_path)
    
    def generate(self,epoch=0,logs=0,num_chars=300,diversities=[0.2, 0.5, 1.0, 1.2]):
        #print()
        #print('----- Generating text after Epoch: %d' % epoch+1)
        start_index = random.randint(0, len(self.text) - self.bptt - 1)
        
        for diversity in diversities:
            print('----- diversity:', diversity)
            generated = ''
            sentence = text[start_index: start_index + self.bptt]
            #generated += sentence
            #generated = ""
            #print('----- Generating with seed: "' + sentence + '"')
            #sys.stdout.write(generated)

            for i in range(num_chars):
                x_pred = np.zeros((1, self.bptt, len(self.chars)))
                for t, char in enumerate(sentence):
                    if(char not in self.char_indices):
                        x_pred[0, t, self.char_indices[" "]] = 1.
                    else:
                        x_pred[0, t, self.char_indices[char]] = 1.

                preds = self.model.predict(x_pred, verbose=0)[0]
                next_index = self.sample(preds, diversity)
                next_char = self.indices_char[next_index]

                generated += next_char
                sentence = sentence[1:] + next_char

                sys.stdout.write(next_char)
                sys.stdout.flush()
            print()
    
    def on_epoch_end(self,epoc,logs):
        #if((self.save_interval != 0) and (epoc % self.save_interval == 0) and self.save_path!=""):
        #    print("Saving model...")
        #    self.save_model(self.save_path)
        if(self.generate_text_while_training):
            self.generate(epoch=epoc,logs=logs)
    
    def train(self,text,validation_split=0.02,
              num_epochs=1,batch_size=32,bptt=10,dim_hidden=128,learning_rate=0.001,
              generate_text=False,save_interval=1,save_path="",log_path="",vocab_save_path=""):
        self.bptt = bptt
        self.dim_hidden = dim_hidden
        self.generate_text_while_training = generate_text
        self.save_interval = save_interval
        self.save_path = save_path
        self.vocab_save_path = vocab_save_path
        callback = LambdaCallback(on_epoch_end=self.on_epoch_end)
        x,y = self.prepare_training_data(text)
        self.model = self.build_model(learning_rate)
        print("Number of parameters in model:",self.model.count_params())
        self.history = self.model.fit(x, y,
          batch_size=batch_size,
          epochs=num_epochs,
          validation_split = validation_split,
          callbacks=[callback, 
                    ModelCheckpoint(save_path, monitor='val_loss', verbose=0, 
                                    save_best_only=True, save_weights_only=False, mode='auto', period=1),
                    EarlyStopping(monitor='val_loss', min_delta=0.001, patience=6, verbose=0, mode='auto'),
                    ReduceLROnPlateau(monitor='val_loss', factor=0.25, patience=1, verbose=0, mode='auto', 
                                      epsilon=0.001, cooldown=0),
                    CSVLogger(log_path, separator=',', append=False)])
        #if(save_interval != 0 and save_path != ""):
        #    self.save_model(save_path)
        return self.history
        
    
    def prepare_test_input(self,text):
        step = 1
        sentences = []
        next_chars = []
        for i in range(0, len(text) - self.bptt, step):
            sentences.append(text[i: i + self.bptt])
            next_chars.append(text[i + self.bptt])
        print('number of sequences:', len(sentences))

        print('Vectorization...')
        x = np.zeros((len(sentences), self.bptt, len(self.chars)), dtype=np.bool)
        y = np.zeros((len(sentences), len(self.chars)), dtype=np.bool)
        for i, sentence in enumerate(sentences):
            for t, char in enumerate(sentence):
                if(char not in self.char_indices.keys()):
                    x[i, t, self.char_indices[" "]] = 1
                else:
                    x[i, t, self.char_indices[char]] = 1
            if(next_chars[i] not in self.char_indices):
                y[i, self.char_indices[" "]] = 1
            else:
                y[i, self.char_indices[next_chars[i]]] = 1
        return x,y
    
    #def evaluate(self,text): # make some arrangement to use LSTM instead of CuDNNLSTM
    #    x,y = self.prepare_test_input(text)
    #    predictions = self.model.predict(x)
    #    probabilities = np.sum(np.multiply(predictions,y),axis=1)
    #    log_prob = np.log(probabilities)
    #    sum_log_prob = np.sum(log_prob)
    #    pp = math.exp(-sum_log_prob/(len(text)-self.bptt))
    #    return pp
    
    def evaluate(self,text): # make some arrangement to use LSTM instead of CuDNNLSTM
        x,y = self.prepare_test_input(text)
        loss = self.model.evaluate(x=x, y=y, batch_size=None, verbose=0, sample_weight=None, steps=None)
        pp = math.exp(loss)
        return pp
    
    #TODO : loading saved models


# In[3]:


class DataUtils:
    def read_file(self,path,encoding='utf-8'):
        data_path = path
        f = codecs.open(data_path, encoding=encoding)
        raw_data = f.read()
        f.close()
        return raw_data
    
    def get_test_train_split(self,text,train_fraction):
        words = text.split()
        train_index = max(1,math.floor(train_fraction*len(words)))
        training_data = " ".join(words[:train_index])
        testing_data = " ".join(words[train_index:])
        return training_data,testing_data
    
    def pickler(self,path,pkl_name,obj):
        with open(os.path.join(path, pkl_name), 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def unpickler(self,path,pkl_name):
        with open(os.path.join(path, pkl_name) ,'rb') as f:
            obj = pickle.load(f)
        return obj


# In[4]:


data_utils = DataUtils()
text = data_utils.read_file("../data/ABrBuCaCh_utf8.txt")
training_data,testing_data = data_utils.get_test_train_split(text,0.98)


# In[5]:


print("Number of characters in data:",len(text))
print("Number of characters in training data:",len(training_data))
print("Number of characters in testing data:",len(testing_data))


# In[6]:


model_save_path = '../saved_models/k3cd05_ABrBuCaCh/{epoch:02d}-{val_loss:.2f}.hdf5'
weight_save_path = '../saved_models/k3cd05_ABrBuCaCh/weights'
log_path = '../saved_models/k3cd05_ABrBuCaCh/log.csv'
vocab_save_path = '../saved_models/k3cd05_ABrBuCaCh/'


# In[7]:


char_lstm = CharLSTM()


# In[ ]:


history = char_lstm.train(text=training_data,validation_split=0.02,
                          num_epochs=80, batch_size=128, bptt=100, dim_hidden=512, learning_rate=0.001,
                          generate_text=False,save_path=model_save_path,log_path=log_path,
                          vocab_save_path=vocab_save_path)


# In[18]:


print(history.history)


# In[12]:


perplexity = char_lstm.evaluate(testing_data)
print("perplexity:{}".format(perplexity))


# In[17]:


char_lstm.generate()


# In[ ]:


# dropout 0.4 : best epoc :16 . loss: 1.1080 - val_loss: 1.1466

# dropout 0.5 : best epoc :16 . loss: 1.1825740571245948 - val_loss: 1.167642938599883 
#callbacks=[callback, 
#                    ModelCheckpoint(save_path, monitor='val_loss', verbose=1, 
#                                    save_best_only=True, save_weights_only=False, mode='auto', period=1),
#                    EarlyStopping(monitor='val_loss', min_delta=0.005, patience=6, verbose=1, mode='auto'),
#                    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, mode='auto', 
#                                      epsilon=0.0003, cooldown=0, min_lr=0.0001),
#                    CSVLogger(log_path, separator=',', append=False)])


