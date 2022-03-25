import pandas as pd
import pickle
import tensorflow as tf
from pandas import concat
from keras.layers import GRU
from pandas import DataFrame
from keras.layers import Dense
from keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

df = pd.read_excel('Abhinav (1).xlsx')
print(df.head)
# df = df.drop(['Date','Hour'], axis=1)
print(df.shape)
print(df)


# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# dataset['Day'] = pd.to_datetime(dataset['Day'])
# df = df.set_index('Date')
values = df.values
encoder = LabelEncoder()
values[:, 1] = encoder.fit_transform(values[:, 1])
values = values.astype(float)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

print(scaled[:, 1])
print(scaled[:, 0])

n_input = 1
n_features = 7

reframed = series_to_supervised(scaled, n_input, 1)
# print(reframed.shape)

# Splitting the dataset
# print(df.columns)
values = reframed.values
train = values[:-2000, 1:]
test = values[-2000:, 1:]

ytrain = values[:-2000, 0]
ytest = values[-2000:, 0]

print(train)

print(test)

print(ytrain)

print(ytest)

n_obs = n_input * n_features
train_X, train_y = train[:, :n_obs], ytrain
test_X, test_y = test[:, :n_obs], ytest
print(train_X.shape, len(train_X), train_y.shape)

train_X = train_X.reshape((train_X.shape[0], n_input, n_features))
test_X = test_X.reshape((test_X.shape[0], n_input, n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

print("Train-x\n", train_X)

print("Train-y\n", train_y)

print("test-x\n", test_X)

print("test-x\n", test_y)

# GRU
model = Sequential()
model.add(GRU(50, return_sequences=True, input_shape=(n_input, n_features)))
model.add(GRU(50, return_sequences=True, input_shape=(n_input, n_features)))
model.add(GRU(50))
model.add(Dense(1))


model.compile(loss='mae', optimizer='adam', metrics=[tf.keras.metrics.RootMeanSquaredError()])
history = model.fit(train_X, train_y, epochs=350, batch_size=72, validation_data=(test_X, test_y), verbose=2,
                    shuffle=False)

with open("model_pickle","wb") as f:
    pickle.dump(model, f)



