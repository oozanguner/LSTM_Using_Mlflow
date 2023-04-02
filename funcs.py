import pandas as pd
import numpy as np
import tensorflow as tf
import datetime as dt
import mlflow
from pydantic import BaseModel, Field
from fastapi import FastAPI
import os
import uvicorn
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import ModelCheckpoint
from keras.metrics import RootMeanSquaredError
from keras.losses import MeanSquaredError
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from warnings import filterwarnings
import time

filterwarnings("ignore")

def preprocessing(file_path, input_target = "Tüketim Miktarý (MWh)", output_target = "Consumption_MWh", date_col_names = ["Tarih","Saat"]):
    dataframe = pd.read_csv(file_path, encoding='unicode_escape')
    dataframe["DATE"] = dataframe.apply(lambda x: pd.to_datetime(x[date_col_names[0]]).strftime("%d-%m-%Y ") + pd.to_datetime(x[date_col_names[1]]).strftime("%H:%M"), axis=1)
    dataframe["DATE"] = pd.to_datetime(dataframe["DATE"])
    dataframe = dataframe.sort_values(by="DATE", ascending=True).drop(date_col_names, axis=1).reset_index(drop=True)
    dataframe.rename(columns={input_target:output_target}, inplace=True)
    dataframe[output_target] = [float (row[0] + row[1] + "." + row[2]) for row in dataframe[output_target].str.extract (r'(\d+)[.](\d+)[,](\d+)', expand=True).values]
    dataframe[output_target] = dataframe[output_target].astype(float)
    df = dataframe[output_target]

    return df

def determine_train_val_test(dataframe, window_size=4, model_weight=0.9, train_weight=0.8):
    sequences = len(dataframe) // window_size

    model_observations = int(model_weight * sequences) * window_size
    model_df, test_df = dataframe[:model_observations], dataframe[model_observations:]

    sequences_model = len(model_df) // window_size
    train_observations = int(train_weight * sequences_model) * window_size

    train_df, val_df = model_df[:train_observations], model_df[train_observations:]

    return train_df, val_df, test_df

def data_to_X_y(dataframe, window_size=4):
    np_df = dataframe.to_numpy()
    X = []
    y = []

    for i in range(len(np_df)- window_size):
        row = [[a] for a in np_df[i:i+window_size]]
        label = np_df[i+window_size]

        X.append(row)
        y.append(label)

    return np.array(X), np.array(y)

def scaling (data):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.reshape(-1, data.shape[-1])).reshape(data.shape)

    return scaled_data

# MODEL
def lstm_model(X_train, y_train, X_val, y_val, n_input=4, n_features=1, epoch=3, batch=32):
    model = Sequential ()
    model.add (LSTM (units=64, activation="relu", input_shape=(n_input, n_features), return_sequences=True))
    model.add (LSTM (units=64, activation="relu", return_sequences=True))
    model.add (LSTM (units=64, activation="relu", return_sequences=False))
    model.add (Dense (1))
    cp = ModelCheckpoint("models/lstm_model", save_best_only=True)
    model.compile (optimizer=Adam(learning_rate=0.01), loss=MeanSquaredError(), metrics=RootMeanSquaredError())
    model.fit (X_train, y_train, validation_data = (X_val, y_val), epochs=epoch, batch_size=batch,shuffle=False, callbacks = [cp])

    return model
