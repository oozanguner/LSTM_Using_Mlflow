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


logged_model = 'file:///Users/ozanguner/VS_Projects/LSTM_Using_Mlflow/mlruns/577697670473209246/88cb5c9a3e974930bffbbc46f3c79fea/artifacts/model'

# Load model as a PyFuncModel.
model = mlflow.pyfunc.load_model(logged_model)

@app.post("/")
async def prediction(items:Inputs):
    data = pd.DataFrame(items.dict().values())
    inp = data.to_numpy().reshape(-1, data.shape[0], data.shape[1])
    y_pred = model.predict (inp)

    return {"Prediction":y_pred}
