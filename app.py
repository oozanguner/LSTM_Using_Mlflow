from funcs import *
from consumption import *

app = FastAPI()

all_models=mlflow.search_runs(experiment_ids="668537897870621134")

best_model_run = (all_models[['run_id','metrics.val_root_mean_squared_error','metrics.root_mean_squared_error']]
 .sort_values(by=["metrics.val_root_mean_squared_error","metrics.root_mean_squared_error"], 
              ignore_index=True)[:1]
              .run_id[0])

logged_model = f"runs:/{best_model_run}/model"

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