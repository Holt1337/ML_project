from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
from fastapi.encoders import jsonable_encoder
from sklearn.linear_model import Ridge
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

cat_dum_cols = ['fuel', 'seller_type', 'transmission', 'owner', 'seats']
colum_list = ['year','km_driven','mileage','engine','max_power','torque','max_torque_rpm','fuel_Diesel','fuel_LPG','fuel_Petrol','seller_type_Individual','seller_type_Trustmark Dealer','transmission_Manual','owner_Fourth & Above Owner','owner_Second Owner','owner_Test Drive Car','owner_Third Owner','seats_4','seats_5','seats_6','seats_7','seats_8','seats_9','seats_10','seats_14']

def get_num(value):
    if type(value) == str:
        value = ''.join([c for c in value if c.isdigit() or c == '.'])
    if value == '':
        value = np.nan
    return value

def get_first_torque(torque):
    if type(torque) == str and torque != '':
        print('XXXXXXXXXXXXXXXXXXX')
        torque = ''.join([c for c in torque.split()[0] if c.isdigit() or c == '.'])
        if torque == '':
            torque = np.nan
    return torque

def get_second_torque(torque):
    if type(torque) == str and torque != '':
        lst = torque.split()
        if len(lst) > 1:
            torque = ''.join([c for c in torque.split()[1] if c.isdigit() or c in '.-'])
            lst2 = torque.split('-')
            if len(lst2) == 2:
                torque = (float(lst2[0]) + float(lst2[1]))/2
        else:
            torque = np.nan
    return torque

def data_proc(object):
    object['mileage'] = object.mileage.apply(get_num)
    object['engine'] = object.engine.apply(get_num)
    object['max_power'] = object.max_power.apply(get_num)

    object['max_power'] = pd.to_numeric(object.max_power)
    object['engine'] = pd.to_numeric(object.engine, downcast='integer')
    object['seats'] = pd.to_numeric(object.seats, downcast='integer')

    object['max_torque_rpm'] = object.torque.apply(get_second_torque)
    object['torque'] = object.torque.apply(get_first_torque)

    object['max_torque_rpm'] = pd.to_numeric(object.max_torque_rpm)
    object['torque'] = pd.to_numeric(object.torque)

    object = pd.get_dummies(object, drop_first=True, columns=cat_dum_cols)
    object_new = pd.DataFrame(object, columns=colum_list)
    object_new = object_new.fillna(value=0)

    scaler = StandardScaler()
    object_new = scaler.fit_transform(object_new)

    return object_new



class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str 
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


def predict(item):
    if type(item) == dict:
        objct = pd.DataFrame([item])
    else:
        objct = item


    objct2 = data_proc(objct)
    decision_model = open('ridge.pkl', 'rb')
    rdg = pickle.load(decision_model)
    decision_model.close()
    pred_rdg = rdg.predict(objct2)
    objct.selling_price = pred_rdg



    return pred_rdg


@app.get("/")
async def get(request: Request, message='ML project'):
    return templates.TemplateResponse('main_page.html',
                                      {"request": request,
                                       "message": message})

@app.post("/predict_item")
def predict_item(item: Item) -> float:
    json_data = jsonable_encoder(item)
    item = json_data
    pred = predict(json_data)
    item['selling_price'] = pred[0]
    return item


@app.post("/predict_items")
def predict_items(items: UploadFile):
    data = pd.read_csv(items.file)
    pred = predict(data)
    data.selling_price = pred
    data.to_csv('output.csv', index=False)

    return items




