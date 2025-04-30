import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from konlpy.tag import Komoran

# data load
train = pd.read_table("/Users/hwang-gyuhan/Desktop/Collage/4-1/자연어처리/Mid/dataset/NSMC/ratings_train.txt")
test = pd.read_table("/Users/hwang-gyuhan/Desktop/Collage/4-1/자연어처리/Mid/dataset/NSMC/ratings_test.txt")
print(f"train shape => {train.shape} \ntest shape => {test.shape}")
train.columns

'''
check data Nan value
1. drop_duplicates
2. dropna
3. Komoran
4. Removing stopwords (particles, punctuation, suffixes)
5. Implement Bag of Words, word to index, and index to word
6. Analyze sentence length distribution and determine an appropriate maximum character length
7. pad_sequences
'''
train.isnull().sum()
test.isnull().sum()

tokenizer = Komoran()

# data preprocessing
def preprocess(train, test):
    train.drop_duplicates(subset=['document'], inplace=True)
    test.drop_duplicates(subset=['document'], inplace=True)
    train = train.dropna()
    test = test.dropna()
    print(f"train shape => {train.shape} \ntest shape => {test.shape}")
    
    train_tokenized = [[token+"/"+POS for token, POS in tokenizer.pos(doc_)] for doc_ in train['document']]
    test_tokenized = [[token+"/"+POS for token, POS in tokenizer.pos(doc_)] for doc_ in test['document']]
    
    exclusion_tags = ['JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ', 'JX', 'JC',
                      'SF', 'SP', 'SS', 'SE', 'SO', 'EF', 'EP', 'EC', 'ETN', 'ETM',
                      'XSN', 'XSV', 'XSA']
    
    f = lambda x: x in exclusion_tags
    
    X_train = []
    for i in range(len(train_tokenized)):
        temp = []
        for j in range(len(train_tokenized[i])):
            if f(train_tokenized[i][j].split('/')[1]) is False:
                temp.append(train_tokenized[i][j].split('/')[0])
        X_train.append(temp)
    
    X_test = []
    for i in range(len(test_tokenized)):
        temp = []
        for j in range(len(test_tokenized[i])):
            if f(test_tokenized[i][j].split('/')[1]) is False:
                temp.append(test_tokenized[i][j].split('/')[0])
        X_test.append(temp)
    
    words = np.concatenate(X_train).tolist()
    counter = Counter(words)
    counter = counter.most_common(30000-4)
    vocab = ['<PAD>', '<BOS>', '<UNK>', '<UNUSED>'] + [key for key, _ in counter]
    word_to_index = {word:index for index, word in enumerate(vocab)}
    
    def wordlist_to_indexlist(wordlist):
        return [word_to_index[word] if word in word_to_index else word_to_index['<UNK>'] for word in wordlist]
    
    X_train = list(map(wordlist_to_indexlist, X_train))
    X_test = list(map(wordlist_to_indexlist, X_test))
    
    return X_train, np.array(list(train['label'])), X_test, np.array(list(test['label'])), word_to_index

X_train, y_train, X_test, y_test, word_to_index = preprocess(train, test)
index_to_word = {index:word for word, index in word_to_index.items()}

all_data = list(X_train)+list(X_test)

num_tokens = [len(tokens) for tokens in all_data]
num_tokens = np.array(num_tokens)


print(f"토큰 길이 평균: {np.mean(num_tokens)}")
print(f"토큰 길이 최대: {np.max(num_tokens)}")
print(f"토큰 길이 표준편차: {np.std(num_tokens)}")

max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)
maxlen = int(max_tokens)
print(f'설정 최대 길이: {maxlen}')
print(f'전체 문장의 {np.sum(num_tokens < max_tokens) / len(num_tokens)}%가 설정값인 {maxlen}에 포함됩니다.')

# Model
X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train,
                                                       padding='pre',
                                                       value=word_to_index["<PAD>"],
                                                       maxlen=70)

X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test,
                                                       padding='pre',
                                                       value=word_to_index["<PAD>"],
                                                       maxlen=70)

vocab_size = 30000
word_vector_dim = 16

model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size, word_vector_dim, input_shape=(None,)))
model.add(tf.keras.layers.Dropout(0.3))  
model.add(tf.keras.layers.LSTM(units=8, return_sequences=False))
model.add(tf.keras.layers.Dropout(0.3))  
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=True,
                                                  stratify=y_train, random_state=777)

lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',       
    factor=0.5,               
    patience=3,               
    verbose=1,                
    min_lr=1e-6               
)

# Model
history = model.fit(
    X_train, y_train,
    epochs=25,
    batch_size=256,
    validation_data=(X_val, y_val),
    verbose=1,
    callbacks=[lr_scheduler] 
)


predict = model.evaluate(X_test, y_test, verbose=1)
print("테스트셋 평가 결과:", predict)
history_dict = history.history

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epochs = range(1, len(acc) + 1)

# graph
fig, axs = plt.subplots(1, 2, figsize=(12, 5))  

axs[0].plot(epochs, loss, 'ro-', label='Training loss')       
axs[0].plot(epochs, val_loss, 'bo-', label='Validation loss')
axs[0].set_title('Training and validation loss')
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Loss')
axs[0].legend()
axs[0].grid(True)

axs[1].plot(epochs, acc, 'ro-', label='Training Accuracy')
axs[1].plot(epochs, val_acc, 'bo-', label='Validation Accuracy')
axs[1].set_title('Training and validation accuracy')
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('Accuracy')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()
