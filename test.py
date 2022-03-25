import pickle
from operator import concat
import pandas as pd
import numpy as np
import tensorflow as tf
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


# arr = list()
# arr = [[5],[38],[2],[5],[0],[0],[0]]
# arr = [5,38,2,5,0,0,0]
arr = np.array([5,38,2,5,0,0,0])
arr = arr.reshape(1,7)
df = pd.DataFrame(list(zip(arr)))
print(df)


values = df.values
encoder = LabelEncoder()
values[:, 0] = encoder.fit_transform(values[:, 0])
values = values.astype(float)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

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

n_input = 2
n_features = 7
reframed = series_to_supervised(scaled, n_input, 1)
values = reframed.values
arr = values[:]


with open("model_pickle", "rb") as f:
    load_model = pickle.load(f)
# load_model = tf.keras.models.load_model('models.pkl')
prediction = load_model.predict(arr)
values = int(np.argmax(prediction))
print(values)

