from funcs import requests
from consumption import Consumptions

def define_inputs():
    inp1 = float(input("First hour energy consumption(mWH): "))
    inp2 = float(input("Second hour energy consumption(mWH): "))
    inp3 = float(input("Third hour energy consumption(mWH): "))
    inp4 = float(input("Fourth hour energy consumption(mWH): "))

    return inp1, inp2, inp3, inp4


def get_get():
    get_url = "http://127.0.0.1:8000"
    message = requests.get(get_url).json()["message"]

    print(message)


def get_prediction(consump1, consump2, consump3, consump4):
    post_url = "http://127.0.0.1:8000/predict"
    inputs = Consumptions(Consumption_0=consump1, Consumption_1=consump2, Consumption_2=consump3, Consumption_3=consump4).dict()
    prediction = requests.post(post_url, json=inputs).json()["prediction"]
    
    print(f"Prediction of the next hour energy comsumption(mWH): {prediction}") 


if __name__ == "__main__":
    print("#" * 50)
    get_get()
    inp1, inp2, inp3, inp4 = define_inputs()
    get_prediction(inp1, inp2, inp3, inp4)
