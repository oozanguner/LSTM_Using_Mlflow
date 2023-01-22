import pandas as pd
import tensorflow as tf
import datetime as dt
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import mlflow
from pathlib import Path
from urllib.parse import urlparse
from warnings import filterwarnings
from chardet import detect

filterwarnings("ignore")

import os

data_path = 'dataset/GercekZamanliTuketim_01012019_16012023.csv'
base_dir = os.getcwd()
file_path = os.path.join(base_dir, data_path)

df = pd.read_csv(file_path, encoding='unicode_escape')

exp_name="EnergyConsumption"
client = mlflow.MlflowClient()
mlflow.set_experiment(exp_name)
mlflow.tensorflow.autolog()


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


train, test = preprocess (df)


# SCALING
def scaling(data=train):
    scaler = MinMaxScaler ()
    scaled_train = scaler.fit_transform (data)
    return scaled_train


scaled_train = scaling (train)


def ts_generator(data=train, targets=train, n_input=24, batch_size=1):
    generator = tf.keras.preprocessing.sequence.TimeseriesGenerator (data=data, targets=targets,
                                                                     length=n_input, batch_size=batch_size)
    return generator


generator = ts_generator (data=scaled_train, targets=scaled_train)


# MODEL
def create_model(n_input=24, n_features=1, generator = generator):
    model = Sequential ()
    model.add (LSTM (units=64, activation="relu", input_shape=(n_input, n_features), return_sequences=True))
    model.add (LSTM (units=64, activation="relu", return_sequences=True))
    model.add (LSTM (units=64, activation="relu", return_sequences=False))
    model.add (Dense (1))
    model.compile (optimizer="adam", loss="mse")
    model.fit (generator, epochs=3, shuffle=False)

    return model

with mlflow.start_run(run_name = "lstm") as run:
    model = create_model()
    tracking_url_type_store = urlparse (mlflow.get_tracking_uri ()).scheme
    mlflow.tensorflow.log_model (model, "model")

    if tracking_url_type_store != "file":
        mlflow.tensorflow.log_model (model, "model", registered_model_name="lstm")
    else:
        mlflow.tensorflow.log_model (model, "model")




# PREDICTION
#first_eval_batch = test[-24:]
#
#sc2 = MinMaxScaler ()
#sc_first_eval = sc2.fit_transform (first_eval_batch)
#y_pred_sc = model.predict (sc_first_eval)
#y_pred = sc2.inverse_transform (y_pred_sc)
#
#pred_df = pd.DataFrame ({"True": first_eval_batch.flatten (),
#                         "Pred": y_pred.flatten ()})
#







