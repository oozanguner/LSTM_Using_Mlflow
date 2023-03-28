import pandas as pd
import tensorflow as tf
import datetime as dt
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import ModelCheckpoint
from keras.metrics import RootMeanSquaredError
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from warnings import filterwarnings
import numpy as np

filterwarnings("ignore")


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

#def ts_generator(data, targets, n_input=3, batch_size=1):
#    generator = tf.keras.preprocessing.sequence.TimeseriesGenerator (data=data, targets=targets,
#                                                                     length=n_input, batch_size=batch_size)
#    return generator

# MODEL
def lstm_model(X_train, y_train, X_val, y_val, n_input=4, n_features=1):
    model = Sequential ()
    model.add (LSTM (units=64, activation="relu", input_shape=(n_input, n_features), return_sequences=True))
    model.add (LSTM (units=128, activation="relu", return_sequences=True))
    model.add (LSTM (units=64, activation="relu", return_sequences=False))
    model.add (Dense (1))
    cp = ModelCheckpoint("/models/lstm_model", save_best_only=True)
    model.compile (optimizer=Adam(learning_rate=0.01), loss=RootMeanSquaredError(), metrics=RootMeanSquaredError())
    model.fit (X_train, y_train, validation_data = (X_val, y_val), epochs=3, shuffle=False, callbacks=[cp])

    return model


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

def determine_train_val_test(dataframe, window_size=4, model_weight=0.9, train_weight=0.8):
    sequences = len(dataframe) // window_size

    model_observations = int(model_weight * sequences) * window_size
    model_df, test_df = dataframe[:model_observations], dataframe[model_observations:]

    sequences_model = len(model_df) // window_size
    train_observations = int(train_weight * sequences_model) * window_size

    train_df, val_df = model_df[:train_observations], model_df[train_observations:]

    return train_df, val_df, test_df

