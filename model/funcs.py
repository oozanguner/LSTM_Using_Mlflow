import pandas as pd
import tensorflow as tf
import datetime as dt
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import mlflow
from mlflow.tracking.client import MlflowClient
from pathlib import Path
from urllib.parse import urlparse
from warnings import filterwarnings
from chardet import detect

filterwarnings("ignore")

import os

def preprocess(dataframe):
    dataframe["Tarih_Saat"] = [row[0] + "." + row[1] for row in dataframe[["Tarih", "Saat"]].values]
    dataframe["Tarih_Saat"] = pd.to_datetime (dataframe["Tarih_Saat"])

    dataframe.rename (columns={"Tüketim Miktarý (MWh)": "TuketimMiktari_MWh"}, inplace=True)
    dataframe["TuketimMiktari_MWh"] = [float (row[0] + row[1] + "." + row[2]) for row in
                                dataframe["TuketimMiktari_MWh"].str.extract (r'(\d+)[.](\d+)[,](\d+)', expand=True).values]
    dataframe.drop (["Tarih", "Saat"], axis=1, inplace=True)
    dataframe.sort_values ("Tarih_Saat", inplace=True)
    dataframe.set_index ("Tarih_Saat", inplace=True)

    today = dt.datetime.today ()

    new_df = dataframe[dataframe.index <= today]

    max_date = new_df.index.max ()
    train_end = max_date - dt.timedelta (days=30)
    test_start = train_end + dt.timedelta (hours=1)

    train = new_df[:train_end].to_numpy ()
    test = new_df[test_start:].to_numpy ()

    return train, test

# SCALING
def scaling(data):
    scaler = MinMaxScaler ()
    scaled_train = scaler.fit_transform (data)
    return scaled_train

def ts_generator(data, targets, n_input=3, batch_size=1):
    generator = tf.keras.preprocessing.sequence.TimeseriesGenerator (data=data, targets=targets,
                                                                     length=n_input, batch_size=batch_size)
    return generator

# MODEL
def create_model(generator, n_input=3, n_features=1):
    model = Sequential ()
    model.add (LSTM (units=64, activation="relu", input_shape=(n_input, n_features), return_sequences=True))
    model.add (LSTM (units=64, activation="relu", return_sequences=True))
    model.add (LSTM (units=64, activation="relu", return_sequences=False))
    model.add (Dense (1))
    model.compile (optimizer="adam", loss="mse")
    model.fit (generator, epochs=3, shuffle=False)

    return model