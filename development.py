from funcs import *
import mlflow

filterwarnings("ignore")

import os
parent_path = 'model'
path = 'dataset/GercekZamanliTuketim_01012019_16012023.csv'
base_dir = os.getcwd()
file_path = os.path.join(base_dir,parent_path,path)

df = pd.read_csv(file_path, encoding='unicode_escape')

feats = {
    "n_input":4,
    "batch_size":3,
    "n_features":1
}

tracking_uri = mlflow.get_tracking_uri()

exp_name="EnergyConsumption"
exper = mlflow.set_experiment(exp_name)
exper_id = exper.experiment_id

train, test = preprocess (dataframe=df)

scaled_train = scaling (data=train)

generator = ts_generator (data=scaled_train, targets=scaled_train, n_input=feats["n_input"], batch_size=feats["batch_size"])

registered_model = "lstm_model"

with mlflow.start_run(run_name = "lstm_energy", experiment_id=exper_id) as run:
    mlflow.tensorflow.autolog()
    model = lstm_model(n_input=feats["n_input"], n_features=feats["n_features"], generator=generator)
    run_id = mlflow.active_run().info.run_id
    artifact_path = "model"

    model_uri = "runs:/{run_id}/{artifact_path}".format(run_id=run_id, artifact_path=artifact_path) 
    model_details = mlflow.register_model(model_uri=model_uri, name=registered_model)
