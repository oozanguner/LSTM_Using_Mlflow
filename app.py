from funcs import *
from consumption import *

app = FastAPI()

logged_model = 'file:///Users/ozanguner/VS_Projects/LSTM_Using_Mlflow/mlruns/668537897870621134/9cc25c3aba34491ca0cd05b73ee25870/artifacts/model'

model_name = "lstm_model"

model = mlflow.pyfunc.load_model(logged_model)

@app.get("/")
async def root():
    return {"message":"Welcome to your Energy Consumption Prediction FastAPI"}

@app.post("/predict", response_model=Predictions)
async def prediction(items:Consumptions):
    data = np.array([[items.Consumption_0, items.Consumption_1, items.Consumption_2, items.Consumption_3]])
    inp = data.reshape(-1, data.shape[1], data.shape[0])
    prediction = model.predict (inp)

    return {"prediction":prediction[0].item()}

if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=8000)
