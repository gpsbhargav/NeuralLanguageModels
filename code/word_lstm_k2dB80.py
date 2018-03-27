
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
from keras.layers import Input,Embedding
from keras.optimizers import RMSprop,SGD,Adam
from keras.utils.data_utils import get_file
from keras.models import load_model
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

import math

import numpy as np
import random
import sys
import io
import os
import string
import dill as pickle
from collections import Counter


import nltk
import codecs


# In[2]:


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

class WordLSTM:
    
    def __init__(self,word_vect_dict,tokenizer_save_path="",embedding_size=300):
        self.word_vect_dict = word_vect_dict
        self.embedding_size = embedding_size
        self.tokenizer_save_path = tokenizer_save_path
    
    def prepare_training_data(self,raw_text):
        maxlen = self.bptt
        step = 5
        EMBEDDING_DIM = self.embedding_size
        embeddings_index = self.word_vect_dict
        text = raw_text.lower()
        to_remove = '!"#$%&\()*+,-/:;<=>?@[\\]^_`{|}~\t\n' + "'"
        text = text.translate(text.maketrans(dict.fromkeys(to_remove)))
        text_without_unk = nltk.wordpunct_tokenize(text)
        #MAX_NUM_WORDS = len(set(text))
        #print("MAX_NUM_WORDS=",MAX_NUM_WORDS)
        text = text_without_unk
        unigram_counter = Counter(nltk.ngrams(text,1))        
        num_unk = 0
        for i in range(len(text_without_unk)):
            #if(  unigram_counter[tuple([text_without_unk[i]])] < 2):
            if(text_without_unk[i] not in self.word_vect_dict or unigram_counter[tuple([text_without_unk[i]])] < 2):
                text[i] = '<unk>'
                num_unk = num_unk + 1
        print("Number of <unk> in training data:{}".format(num_unk))
        sentences = []
        next_words = []
        for i in range(0, len(text) - maxlen, step):
            sentences.append(text[i: i + maxlen])
            next_words.append(text[i + maxlen])
        tokenizer = Tokenizer(num_words=None,
                                       filters='',
                                       lower=False,
                                       split=" ",
                                       char_level=False,
                                       oov_token='<unk>')
        tokenizer.fit_on_texts(sentences)
        #necessary because keras word to index dict starts from 1 instead of 0. 
        #It makes to_categorical make an additional category that is never used
        tokenizer.word_index[max(tokenizer.word_index, key=tokenizer.word_index.get)] = 0
        self.seq_list = tokenizer.texts_to_sequences(sentences)
        sequences = np.array(tokenizer.texts_to_sequences(sentences))
        # prepare embedding matrix
        num_words = len(tokenizer.word_index)
        print("num_words:",num_words)
        embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
        for word, i in tokenizer.word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
        targets = tokenizer.texts_to_sequences(next_words)
        targets = np.array(targets)
        #targets = np.array(to_categorical(np.asarray(targets),num_classes=num_words))
        #print(targets)
        print("training sequence shape:",sequences.shape)
        #print("targets shape:",targets.shape)
        self.tokenizer = tokenizer
        self.num_words = num_words
        self.embedding_matrix = embedding_matrix
        #self.MAX_NUM_WORDS = MAX_NUM_WORDS
        self.text = text
        if(self.tokenizer_save_path != ""):
            data_util = DataUtils()
            data_util.pickler(self.tokenizer_save_path,"tokenizer.pkl",self.tokenizer)
        self.sequence = sequences
        self.targets = targets
        return sequences,targets
    
    def build_model(self,learning_rate,dropout):
        model = Sequential()
        model.add(Embedding(self.num_words,
                            self.embedding_size,
                            weights=[self.embedding_matrix],
                            input_length=self.bptt,
                            trainable=False))
        model.add(Dropout(dropout))
        #model.add(CuDNNLSTM(self.dim_hidden,return_sequences=True))
        #model.add(Dropout(dropout))
        model.add(CuDNNLSTM(self.dim_hidden,return_sequences=True))
        model.add(Dropout(dropout))
        model.add(CuDNNLSTM(self.dim_hidden))
        model.add(Dropout(dropout))
        model.add(Dense(self.num_words))
        model.add(Activation('softmax'))
        #optimizer = RMSprop(lr=learning_rate,clipvalue=0.25)
        optimizer = Adam(lr=learning_rate,clipvalue=0.25)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer)
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
    
    def generate(self,epoch=0,logs=0,num_chars=40,diversities=[0.2, 0.5, 1.0, 1.2]):
        #char_indices = dict((c, i) for i, c in enumerate(chars))
        indices_word = dict((i, c) for i, c in enumerate(self.tokenizer.word_index))
        #print()
        #print('----- Generating text after Epoch: %d' % epoch+1)
        start_index = random.randint(0, len(self.text) - self.bptt - 1)
        
        for diversity in diversities:
            print('----- diversity:', diversity)
            generated = ''
            sentence = self.text[start_index: start_index + self.bptt]
            #print(sentence)
            #generated += " ".join(sentence)
            generated = ""
            print('----- Generating with seed: "' + " ".join(sentence) + '"')
            #sys.stdout.write(generated)

            for i in range(num_chars):
                #print("Iteration:",i)
                #x_pred = np.zeros((1, self.bptt, len(self.num_words)))
                #for t, char in enumerate(sentence):
                #    x_pred[0, t, self.char_indices[char]] = 1.
                #print(sentence)
                x_pred = np.array(self.tokenizer.texts_to_sequences([sentence]))
                preds = self.model.predict(x_pred, verbose=0)[0]
                next_index = self.sample(preds, diversity)
                next_char = indices_word[next_index] #word indices start from 1
                #print("next_char={},type={}".format(next_char,type(next_char)))
                generated += " " + next_char
                sentence = sentence[1:]
                sentence.append(next_char)
                #print(sentence)
                #sys.stdout.write(next_char)
                #sys.stdout.flush()
            print(generated)
    
    def on_epoch_end(self,epoc,logs):
        #if((self.save_interval != 0) and (epoc % self.save_interval == 0) and self.save_path!=""):
        #    print("Saving model...")
        #    self.save_model(self.save_path)
        if(self.generate_text_while_training):
            self.generate(epoch=epoc,logs=logs)
    
    def train(self,text,validation_split=0.02,
              num_epochs=1,batch_size=32,bptt=10,dim_hidden=128,learning_rate=0.001,
              generate_text=False,dropout=0.00001,save_interval=1,save_path="",log_path=""):
        self.bptt = bptt
        self.dim_hidden = dim_hidden
        self.generate_text_while_training = generate_text
        self.save_interval = save_interval
        self.save_path = save_path
        callback = LambdaCallback(on_epoch_end=self.on_epoch_end)
        x,y = self.prepare_training_data(text)
        self.model = self.build_model(learning_rate,dropout=dropout)
        print("Number of parameters in model:",self.model.count_params())
        callbacks = [callback, 
                    ModelCheckpoint(save_path, monitor='val_loss', verbose=0, 
                                    save_best_only=True, save_weights_only=False, mode='auto', period=1),
                    EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, verbose=0, mode='auto'),
                    ReduceLROnPlateau(monitor='val_loss', factor=0.25, patience=1, verbose=0, mode='auto', 
                                      epsilon=0.001, cooldown=0),
                    CSVLogger(log_path, separator=',', append=False)]
        self.history = self.model.fit(x, y,
          batch_size=batch_size,
          epochs=num_epochs,
          validation_split = validation_split,
          callbacks=callbacks)
        #if(save_interval != 0 and save_path != ""):
        #    self.save_model(save_path)
        return self.history
        
    
    def prepare_test_input(self,raw_text):
        maxlen = self.bptt
        step = 1
        #MAX_NUM_WORDS = self.MAX_NUM_WORDS
        EMBEDDING_DIM = self.embedding_size
        embeddings_index = self.word_vect_dict
        text = raw_text.lower()
        to_remove = '!"#$%&\()*+,-/:;<=>?@[\\]^_`{|}~\t\n' + "'"
        text = text.translate(text.maketrans(dict.fromkeys(to_remove)))
        text = nltk.wordpunct_tokenize(text)
        sentences = []
        next_words = []
        for i in range(0, len(text) - maxlen, step):
            sentences.append(text[i: i + maxlen])
            next_words.append(text[i + maxlen])
        tokenizer = self.tokenizer
        #tokenizer.fit_on_texts(sentences)
        sequences = np.array(tokenizer.texts_to_sequences(sentences))
        targets = tokenizer.texts_to_sequences(next_words)
        targets = np.array(targets)
        #targets = np.array(to_categorical(np.asarray(targets),num_classes=self.num_words))
        #print(targets)
        print("testing sequence shape:",sequences.shape)
        #print("targets shape:",targets.shape)
        return sequences,targets
    
    
    def evaluate(self,text): # make some arrangement to use LSTM instead of CuDNNLSTM
        x,y = self.prepare_test_input(text)
        loss = self.model.evaluate(x=x, y=y, batch_size=None, verbose=0, sample_weight=None, steps=None)
        pp = math.exp(loss)
        #predictions = self.model.predict(x)
        #probabilities = np.sum(np.multiply(predictions,y),axis=1)
        #log_prob = np.log(probabilities)
        #sum_log_prob = np.sum(log_prob)
        #pp = math.exp(-sum_log_prob/(len(text)-self.bptt))
        return pp
    
    #TODO : loading saved models


# In[3]:


data_utils = DataUtils()
text = data_utils.read_file("../data/chunk_shuffled_all.txt")
training_data,testing_data = data_utils.get_test_train_split(text,0.95)

print("Number of words in training data:{}".format(len(training_data.split())))
print("Number of words in testing data:{}".format(len(testing_data.split())))


# In[4]:


model_save_path = '../saved_models/k2wd05U2B80chunkShuffledRaw/{epoch:02d}-{val_loss:.2f}.hdf5'
weight_save_path = '../saved_models/k2wd05U2B80chunkShuffledRaw/weights'
log_path = '../saved_models/k2wd05U2B80chunkShuffledRaw/log.csv'
tokenizer_save_path = '../saved_models/k2wd05U2B80chunkShuffledRaw/'


# In[5]:


pkl_dir = "../embeddings/"
pkl_name = "glove.840B.300d.pickle"
word_vect_dict = data_utils.unpickler(pkl_dir,pkl_name)


# In[6]:


word_lstm = WordLSTM(word_vect_dict,tokenizer_save_path=tokenizer_save_path)


# In[ ]:


history = word_lstm.train(text=training_data,validation_split=0.05,
                          num_epochs=40, batch_size=20, bptt=80, dim_hidden=512, learning_rate=0.001,
                          generate_text=False,dropout=0.5,save_path=model_save_path,log_path=log_path)


# In[ ]:


print(history.history)


# In[ ]:


perplexity = word_lstm.evaluate(testing_data)
print("perplexity:{}".format(perplexity))


# In[ ]:


word_lstm.generate()


# training_data,validation_data = data_utils.get_test_train_split(training_data,0.95)

# with open('train.txt', 'w',encoding='utf8') as the_file:
#     the_file.write(training_data)

# with open('valid.txt', 'w',encoding='utf8') as the_file:
#     the_file.write(validation_data)

# with open('test.txt', 'w',encoding='utf8') as the_file:
#     the_file.write(testing_data)
