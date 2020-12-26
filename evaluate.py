import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

df = pd.read_csv('data.csv')
data = df.drop(columns=['duration_ms', 'popularity', 'key', 'instrumentalness',
                        'valence', 'artists', 'id', 'name', 'release_date', 'mode'])
X = data.drop(columns=['explicit'])
y = data['explicit']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# define the keras model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X_train, y_train, epochs=150, batch_size=100)
# evaluate the keras model
_, accuracy = model.evaluate(X_test, y_test)
print('Accuracy: %.2f' % (accuracy*100))
