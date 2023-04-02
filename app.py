from funcs import *
from consumption import Consumptions

app = FastAPI()

logged_model = 'file:///Users/ozanguner/VS_Projects/LSTM_Using_Mlflow/mlruns/182815131481226371/a3ff4f453d284042b7888b299df0c270/artifacts/model'

model_name = "lstm_model"

# Load model as a PyFuncModel.
model = mlflow.pyfunc.load_model(logged_model)

@app.post("/predict")
async def prediction(items:Consumptions):
    data = pd.DataFrame(items.dict().values())
    inp = data.to_numpy().reshape(-1, data.shape[0], data.shape[1])
    y_pred = model.predict (inp)

    return {"Prediction":y_pred[0].item()}

if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=8000)
