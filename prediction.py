from funcs import *
import mlflow
import os

logged_model = 'file:///Users/ozanguner/VS_Projects/LSTM_Using_Mlflow/mlruns/577697670473209246/da022765022844ed8b83869277c560fb/artifacts/model'

# Load model as a PyFuncModel.
model = mlflow.pyfunc.load_model(logged_model)

# PREDICTION
parent_path = 'model'
path = 'dataset/GercekZamanliTuketim_01012019_16012023.csv'
base_dir = os.getcwd()
file_path = os.path.join(base_dir, parent_path, path)

df = pd.read_csv(file_path, encoding='unicode_escape')
df["DATE"] = df.apply(lambda x: pd.to_datetime(x["Tarih"]).strftime("%d-%m-%Y ") + pd.to_datetime(x["Saat"]).strftime("%H:%M"), axis=1)
df["DATE"] = pd.to_datetime(df["DATE"])
df2 = df.sort_values(by="DATE", ascending=True).drop(["Tarih","Saat"], axis=1).reset_index(drop=True)
df2.rename(columns={"Tüketim Miktarý (MWh)":"Consumption_MWh"}, inplace=True)
df2["Consumption_MWh"] = [float (row[0] + row[1] + "." + row[2]) for row in df2["Consumption_MWh"].str.extract (r'(\d+)[.](\d+)[,](\d+)', expand=True).values]
df2["Consumption_MWh"] = df2["Consumption_MWh"].astype(float)

temp = df2["Consumption_MWh"]

train_df, val_df, test_df = determine_train_val_test(temp)

X_train, y_train = data_to_X_y(train_df)
X_val, y_val = data_to_X_y(val_df)

scaler = MinMaxScaler ()

X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
X_val = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)

y_pred_sc = model.predict (X_train)
y_pred = scaler.inverse_transform (y_pred_sc)

pred_df = pd.DataFrame ({"True": y_train.flatten(),"Pred": y_pred.flatten ()})

y_pred_val_sc = model.predict (X_val)
y_pred_val = scaler.inverse_transform (y_pred_val_sc)

val_pred_df = pd.DataFrame ({"True": y_val.flatten(),"Pred": y_pred_val.flatten ()})

print(val_pred_df)


