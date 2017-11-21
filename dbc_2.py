import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Merge
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.layers import Dropout

# fix random seed for reproducibility
numpy.random.seed(7)


# load the dataset but only keep the top n words, zero the rest
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

def dividir_classe(x, y):
    # divide os dados em DUAS classes
    x1_features = []
    y1_label = []
    x2_features = []
    y2_label = []
    for i in range(0,len(y)):
        if y[i]==1:
            x1_features.append(x[i])
            y1_label.append(y[i])
        else:
            x2_features.append(x[i])
            y2_label.append(y[i])
    return numpy.asarray(x1_features), numpy.asarray(y1_label), numpy.asarray(x2_features), numpy.asarray(y2_label)
	

# truncate and pad input sequences
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

x_train1, y_train1, x_train2, y_train2 =  dividir_classe(X_train, y_train)
x_test1, y_test1, x_test2, y_test2 =  dividir_classe(X_test, y_test)

# create the MODEL 1
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dropout(0.2))
model.add(Dense(1, activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(x_train1, y_train1, epochs=1, batch_size=64)

# create the MODEL 2
#embedding_vecor_length = 32
model2 = Sequential()
model2.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model2.add(Dropout(0.2))
model2.add(LSTM(100))
model2.add(Dropout(0.2))
model2.add(Dense(1, activation='softmax'))
model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model2.summary())
model2.fit(x_train2, y_train2, epochs=1, batch_size=64)

#MODEL 3
embedding_vecor_length = 32
model3 = Sequential()
model3.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model3.add(Dropout(0.2))

model3.add(LSTM(100, return_sequences=True))
model3.layers[2] = model.layers[2]
model3.add(Dropout(0.2))
model3.add(Dense(1, activation='sigmoid',weights=model.layers[-1].get_weights()))

model3.add(LSTM(100))
model3.layers[5] = model2.layers[2]
model3.add(Dropout(0.2))
model3.add(Dense(1, activation='sigmoid',weights=model2.layers[-1].get_weights()))

model3.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
print(model3.summary())
model3.fit(X_train, y_train, epochs=3, batch_size=64)


# Final evaluation of the model TESTE COMPLETO
scores = model3.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))