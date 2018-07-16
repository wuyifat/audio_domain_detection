import pandas as pd
from nltk.tokenize import word_tokenize
import nltk
import string

from collections import Counter
import itertools

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import *
from keras.callbacks import CSVLogger

from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.model_selection import train_test_split


pos_df = pd.read_csv('data/mozilla/domain_food_text_aug_eat.txt', header=None, names=['text'])
neg_df = pd.read_csv('data/mozilla/domain_food_other_text.txt', header=None, names=['text'])
pos_df['label'] = 1
neg_df['label'] = 0
neg_df = neg_df.head(pos_df.shape[0])
df = pd.concat([pos_df, neg_df])
df = df.sample(frac=1).reset_index(drop=True)
train_data, val_data = train_test_split(df, train_size=0.8, random_state=111)


max_vocab = 20000
maxlen = 20
batch_size = 100

def low(s):
    return s.lower()

def remove_punc(s):
    return "".join(c for c in s if c not in string.punctuation)

def lem(tokens):
    lemmatizer = nltk.WordNetLemmatizer()
    lem_tokens = []
    for token in tokens:
        lem_tokens.append(lemmatizer.lemmatize(token))
    return lem_tokens

def prepare_features(df):
    df = df.sample(frac=1).reset_index(drop=True)
    df['token'] = df['text'].apply(low).apply(remove_punc).apply(word_tokenize)
    df['token'] = df['token'].apply(lem)
    
    sentences = []
    for t in df['token']:
        sentences.append(t)
    word_counts = Counter(itertools.chain(*sentences))
    voc_common = [x[0] for x in word_counts.most_common(max_vocab)]
    voc = {x : i+1 for i,x in enumerate(voc_common)}
    
    X = np.array([[voc.get(word, 0) for word in sentence] for sentence in sentences])
    X = sequence.pad_sequences(X, maxlen = maxlen)
    
    y = df['label']
    
    return X, y

def model(nb_epoch=50):
    X_train, y_train = prepare_features(train_data)
    X_test, y_test = prepare_features(val_data)
    
    model = Sequential()
    model.add(Embedding(max_vocab+1, 128, input_length = maxlen))
    model.add(LSTM(128))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    model.compile(loss='binary_crossentropy',optimizer='nadam', metrics=['accuracy'])
    csv_logger = CSVLogger('ckpt/text/log.csv')
    model.fit(X_train, y_train, batch_size = batch_size, nb_epoch = nb_epoch, validation_data = (X_test, y_test), verbose = 1, callbacks=[csv_logger])

    y_test_pred = model.predict(X_test, verbose = 0)

    score, acc = model.evaluate(X_test, y_test,
                            batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)

    acc = accuracy_score(y_test, (y_test_pred > 0.5).astype(int))
    recall = recall_score(y_test, (y_test_pred > 0.5).astype(int))
    print( 'Accuracy: %.2f, Recall %.2f' % (acc, recall))

if __name__ == '__main__':
	model()