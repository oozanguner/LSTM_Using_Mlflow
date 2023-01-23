from model.funcs import *

logged_model = 'file:///Users/ozanguner/VS_Projects/LSTM_Using_Mlflow/model/mlruns/365411092557517721/09d27e7c882145769389933f72899bfc/artifacts/model'

# Load model as a PyFuncModel.
model = mlflow.pyfunc.load_model(logged_model)

# PREDICTION
parent_path = 'model'
path = 'dataset/GercekZamanliTuketim_01012019_16012023.csv'
base_dir = os.getcwd()
file_path = os.path.join(base_dir, parent_path, path)

df = pd.read_csv(file_path, encoding='unicode_escape')
train, test = preprocess (df)
first_eval_batch = test[-3:]

sc2 = MinMaxScaler ()
sc_first_eval = sc2.fit_transform (first_eval_batch)
y_pred_sc = model.predict (sc_first_eval)
y_pred = sc2.inverse_transform (y_pred_sc)

pred_df = pd.DataFrame ({"True": first_eval_batch.flatten (),
                         "Pred": y_pred.flatten ()})


