from fastapi import FastAPI
from pydantic import BaseModel, Field
import mlflow
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

app = FastAPI()

class Inputs(BaseModel):
    Consumption_0 : float=Field(11000,gt=10000, lt=60000)
    Consumption_1 : float=Field(12000,gt=10000, lt=60000)
    Consumption_2 : float=Field(13000,gt=10000, lt=60000)
    Consumption_3 : float=Field(14000,gt=10000, lt=60000)


logged_model = 'file:///Users/ozanguner/VS_Projects/LSTM_Using_Mlflow/mlruns/577697670473209246/da022765022844ed8b83869277c560fb/artifacts/model'

# Load model as a PyFuncModel.
model = mlflow.pyfunc.load_model(logged_model)

@app.post("/")
async def prediction(items:Inputs):
    sc = MinMaxScaler ()
    inp = pd.DataFrame(items.dict().values()).values
    sc_first_eval = sc.fit_transform (inp)
    y_pred_sc = model.predict (sc_first_eval)
    y_pred = sc.inverse_transform (y_pred_sc).flatten().tolist()
    pred_count = len(y_pred)
    forecasts = [f"Forecast_{c}" for c in range(pred_count)]
    pred_dict = dict(zip(forecasts,y_pred))

    return pred_dict
