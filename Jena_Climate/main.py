import os
import datetime
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Data PreProcessing

df = pd.read_csv('/Users/hwang-gyuhan/Desktop/Collage/4-1/자연어처리/Mid/dataset/Jena_Climate/jena_climate_2009_2016.csv')

numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()
ax = sns.heatmap(corr, 
                 vmin=-1, vmax=1, center=0,
                 cmap=sns.diverging_palette(20, 220, n=200),
                 square=True)

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

for col in ['wv (m/s)', 'max. wv (m/s)']:
    df[col] = df[col].replace(-9999.00, 0)
    
features = [list(df.columns)[c] for c in [1,2,6,8,9,11,12]]
df_filt = df[['Date Time']+features].copy()
date_time = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')

timestamp_s = date_time.map(datetime.datetime.timestamp)
day = 24*60*60
year = (365.2425)*day

df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

split_fraction = 0.75075
train_split = int(split_fraction * int(df.shape[0]))

step = 6
past = 720
future = 72
batch_size = 256

data_mean = df[:train_split].mean(axis=0)
data_std = df[:train_split].std(axis=0)

df = (df-data_mean) / data_std
df = df.values
df = pd.DataFrame(df)
df.head()

train_data = df.loc[:train_split-1]
val_data = df.loc[train_split:]

start = past + future
end = start + train_split

x_train = train_data[[i for i in range(11)]].values
y_train = df.iloc[start:end][[1]]

sequence_length = int(past / step)

print('X_train shape == {}.'.format(x_train.shape))
print('y_train shape == {}.'.format(y_train.shape))

dataset_train = keras.preprocessing.timeseries_dataset_from_array(
    x_train.astype(np.float32),
    y_train.astype(np.float32),
    sequence_length=sequence_length,
    sampling_rate=step,
    batch_size=batch_size,
)

x_end = len(val_data) - past - future

label_start = train_split + past + future

x_val = val_data.iloc[:x_end][[i for i in range(11)]].values
y_val = df.iloc[label_start:][[1]]

dataset_val = keras.preprocessing.timeseries_dataset_from_array(
    x_val.astype(np.float32),
    y_val.astype(np.float32),
    sequence_length=sequence_length,
    sampling_rate=step,
    batch_size=batch_size,
)


for batch in dataset_train.take(1):
    inputs, targets = batch

print("Input shape:", inputs.numpy().shape)
print("Target shape:", targets.numpy().shape)
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Model
learning_rate = 0.001
inputs = keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2]))
lstm_out = keras.layers.LSTM(48)(inputs)
outputs = keras.layers.Dense(1)(lstm_out)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
model.summary()


early_stopper = EarlyStopping(monitor="val_loss", min_delta=0, patience=7, restore_best_weights=True)
checkpointer = ModelCheckpoint(monitor="val_loss", save_freq="epoch",filepath="{epoch:03d}-{val_loss:.5f}.h5", verbose=1, save_weights_only=True, save_best_only=True)
lr_reducer = ReduceLROnPlateau(monitor="val_loss", factor=0.169, patience=3, verbose=1)

history = model.fit(
    dataset_train,
    epochs=50,
    validation_data=dataset_val,
    callbacks=[early_stopper, checkpointer, lr_reducer],
)