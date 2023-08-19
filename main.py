import pandas as pd
import datetime
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.preprocessing import MinMaxScaler
from pickle import load

app = FastAPI()


with open('random_forest_model-0.1.0.pkl', 'rb') as file:
    model = load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = load(file)


columns = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'P11', 'DayOfWeek_0', 'DayOfWeek_1', 'DayOfWeek_2', 'DayOfWeek_3', 'DayOfWeek_4', 'DayOfWeek_5', 'DayOfWeek_6', 
           'Month_1', 'Month_2', 'Month_3', 'Month_4', 'Month_5', 'Month_6', 'Month_7', 'Month_8', 'Month_9', 'Month_10', 'Month_11', 'Month_12' ]    

columns_to_scale = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'P11']
columns_nonScaled = ['DayOfWeek_0', 'DayOfWeek_1', 'DayOfWeek_2', 'DayOfWeek_3', 'DayOfWeek_4', 'DayOfWeek_5', 'DayOfWeek_6', 
           'Month_1', 'Month_2', 'Month_3', 'Month_4', 'Month_5', 'Month_6', 'Month_7', 'Month_8', 'Month_9', 'Month_10', 'Month_11', 'Month_12']

df = pd.DataFrame(columns = columns)


class Input(BaseModel):
    year: int
    month: int
    day: int
    P1: float
    P2: float
    P3: float
    P4: float
    P5: float
    P6: float
    P7: float
    P8: float
    P9: float
    P10: float
    P11: float


#  Create predict endpoint
@app.get("/")
async def root(body: Input):
    global df
    result = process(body)
    df = df._append(result,ignore_index=True) #type: ignore
    print()
    scaled_df = pd.DataFrame(scaler.transform(df[columns_to_scale]) , columns = columns_to_scale)
    combined_df = pd.concat([scaled_df , df[columns_nonScaled]] , axis = 1)
    result = model.predict(combined_df)
    
    
    return {'prediction' : result[0]}
    


def process(body):
    processed_body = {
        "P1": body.P1,
        "P2": body.P2,
        "P3": body.P3,
        "P4": body.P4,
        "P5": body.P5,
        "P6": body.P6,
        "P7": body.P7,
        "P8": body.P8,
        "P9": body.P9,
        "P10": body.P10,
        "P11": body.P11,
        "Month_1": 0,
        "Month_2": 0,
        "Month_3": 0,
        "Month_4": 0,
        "Month_5": 0,
        "Month_6": 0,
        "Month_7": 0,
        "Month_8": 0,
        "Month_9": 0,
        "Month_10": 0,
        "Month_11": 0,
        "Month_12": 0,
        "DayOfWeek_0": 0,
        "DayOfWeek_1": 0,
        "DayOfWeek_2": 0,
        "DayOfWeek_3": 0,
        "DayOfWeek_4": 0,
        "DayOfWeek_5": 0,
        "DayOfWeek_6": 0,
    }

    day = datetime.datetime(body.year, body.month, body.day).weekday()
    processed_body[f"DayOfWeek_{day}"] = 1
    processed_body[f"Month_{body.month}"] = 1

    return processed_body
