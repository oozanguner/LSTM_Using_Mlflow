from fastapi import FastAPI
from pydantic import BaseModel
import mlflow

app = FastAPI()

class Inputs(BaseModel):
    Consumption_0 :     float 
    Consumption_1 :     float
    Consumption_2 :     float
    Consumption_3 :     float
    Consumption_4 :     float
    Consumption_5 :     float
    Consumption_6 :     float
    Consumption_7 :     float
    Consumption_8 :     float
    Consumption_9 :     float
    Consumption_10 :    float
    Consumption_11 :    float
    Consumption_12 :    float
    Consumption_13 :    float
    Consumption_14 :    float
    Consumption_15 :    float
    Consumption_16 :    float
    Consumption_17 :    float
    Consumption_18 :    float
    Consumption_19 :    float
    Consumption_20 :    float
    Consumption_21 :    float
    Consumption_22 :    float
    Consumption_23 :    float
    Consumption_24 :    float


logged_model = 'file:///Users/ozanguner/VS_Projects/LSTM_Using_Mlflow/mlruns/577697670473209246/2360e384b95148da980a5708edb0510e/artifacts/model'

# Load model as a PyFuncModel.
model = mlflow.pyfunc.load_model(logged_model)


@app.post("/inputs")
async def get_item_id(items:Inputs):
    return items