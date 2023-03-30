from funcs import *
import mlflow
import os
import time

filterwarnings("ignore")


parent_path = 'model'
path = 'dataset/GercekZamanliTuketim_01012019_16012023.csv'
base_dir = os.getcwd()
file_path = os.path.join(base_dir,parent_path,path)

print("Development Process Starts")

start_prep = time.time()
df = preprocessing(file_path=file_path, input_target="Tüketim Miktarý (MWh)", output_target = "Consumption_MWh", date_col_names = ["Tarih","Saat"])

train_df, val_df, test_df = determine_train_val_test(df,window_size=4, model_weight=0.9, train_weight=0.8)

X_train, y_train = data_to_X_y(train_df)
X_val, y_val = data_to_X_y(val_df)
end_prep = time.time()
print(f"""Preprocessing: {round(end_prep-start_prep,2)} seconds""")

tracking_uri = mlflow.get_tracking_uri()

exp_name="EnergyConsumption"
exper = mlflow.set_experiment(exp_name)
exper_id = exper.experiment_id

registered_model = "lstm_model"

feats = {
    "n_input":4,
    "batch_size":3,
    "n_features":1
}

start_model = time.time()
with mlflow.start_run(run_name = "lstm_energy", experiment_id=exper_id) as run:
    mlflow.tensorflow.autolog()
    model = lstm_model(X_train, y_train, X_val, y_val, n_input=feats["n_input"], n_features=feats["n_features"], epoch = 8)
    run_id = mlflow.active_run().info.run_id
    artifact_path = "model"

    model_uri = "runs:/{run_id}/{artifact_path}".format(run_id=run_id, artifact_path=artifact_path) 
    model_details = mlflow.register_model(model_uri=model_uri, name=registered_model)
end_model = time.time()
print(f"""Model: {end_model-start_model} seconds""")
print("Development Process Completed")



