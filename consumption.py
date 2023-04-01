from funcs import *

class Consumptions(BaseModel):
    Consumption_0 : float=Field(11000,gt=10000, lt=60000)
    Consumption_1 : float=Field(12000,gt=10000, lt=60000)
    Consumption_2 : float=Field(13000,gt=10000, lt=60000)
    Consumption_3 : float=Field(14000,gt=10000, lt=60000)