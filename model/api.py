from fastapi import FastAPI
from enum import Enum
import pydantic

app = FastAPI()

query_list = [
    {"item_name":"first"}, 
    {"item_name":"second"}, 
    {"item_name":"third"}
    ]

class ParamName(str, Enum):
    name = "ozan"


@app.get("/greetings/{isim}")
async def hi(isim:str):
    return {"message":f"Hello {isim}"}

@app.get("/query")
async def get_query(start:int, stop:int):
    return query_list[start:stop]

@app.get("/item/{item_id}")
async def get_item_id(item_id:int):
    return {"item_id":item_id}