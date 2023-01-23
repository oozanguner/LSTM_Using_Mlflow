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

path = 'dataset/GercekZamanliTuketim_01012019_16012023.csv'
base_dir = os.getcwd()
file_path = os.path.join(base_dir, path)

df = pd.read_csv(file_path, encoding='unicode_escape')

tracking_uri = mlflow.get_tracking_uri()

exp_name="EnergyConsumption"
exper = mlflow.set_experiment(exp_name)
exper_id = exper.experiment_id

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

def ts_generator(data=train, targets=train, n_input=3, batch_size=1):
    generator = tf.keras.preprocessing.sequence.TimeseriesGenerator (data=data, targets=targets,
                                                                     length=n_input, batch_size=batch_size)
    return generator


generator = ts_generator (data=scaled_train, targets=scaled_train)

# MODEL
def create_model(n_input=3, n_features=1, generator=generator):
    model = Sequential ()
    model.add (LSTM (units=64, activation="relu", input_shape=(n_input, n_features), return_sequences=True))
    model.add (LSTM (units=64, activation="relu", return_sequences=True))
    model.add (LSTM (units=64, activation="relu", return_sequences=False))
    model.add (Dense (1))
    model.compile (optimizer="adam", loss="mse")
    model.fit (generator, epochs=3, shuffle=False)

    return model

registered_model = "lstm_model"

with mlflow.start_run(run_name = "lstm_energy", experiment_id=exper_id) as run:
    mlflow.tensorflow.autolog()
    model = create_model()
    run_id = mlflow.active_run().info.run_id
    artifact_path = "model"

    model_uri = "runs:/{run_id}/{artifact_path}".format(run_id=run_id, artifact_path=artifact_path) 
    model_details = mlflow.register_model(model_uri=model_uri, name=registered_model)




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







