import pickle
import numpy as np
import tensorflow as tf
from pydantic import BaseModel
from fastapi.testclient import TestClient
from fastapi import FastAPI, Form, Depends
from starlette.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI()
templates = Jinja2Templates(directory="templates")
model = pickle.load(open("models.pkl", "rb"))


@app.get("/predict", response_class=HTMLResponse)
def get_input():
    return '''
    <form method= "post'>
    <title>Visiblity Forecasting</title>
    <legend>Visiblity Forecasting Form </legend><br>
    <br>
    <div class="col-sm-4">
    <label for="Temperature">Temperature</label>
    <input type="number" class="form-control" id="Temperature" name="Temperature" required>
    </div>
    <br>
    
    <br>
    <div class="col-sm-3">
    <label for="Humidity">Humidity</label>
    <input type="number" class="form-control" id="Humidity" name="Humidity" required>
    </div>
    <br>
    <div class="col-sm-3">
    <label for="Wind Speed">Wind Speed</label>
    <input type="number" class="form-control" id="Wind Speed" name="Wind Speed" required>
    </div>
    <br>
    <div class="col-sm-3">
    <label for="Dev Point Temperature">Dev Point Temperature</label>
    <input type="number" class="form-control" id="Dev Point Temperature" name="Dev Point Temperature" required>
    </div>
    <br>
    <div class="col-sm-3">
    <label for="Solar Radiation">Solar Radiation</label>
    <input type="number" class="form-control" id="Solar Radiation" name="Solar Radiation" required>
    </div>
    <br>
    <div class="col-sm-3">
    <label for="Rainfall">Rainfall</label>
    <input type="number" class="form-control" id="Rainfall" name="Rainfall" required>
    </div>
    <br>
    <div class="col-sm-3">
    <label for="Snowfall">Snowfall</label>
    <input type="number" class="form-control" id="Snowfall" name="Snowfall" required>
    </div>
    <br>
    <br>
    <div class="form-group">
    <input class="btn btn-primary" type="submit" value="Result">
    </div>
    <br>
    <br>
    '''


def form_body(cls):
    cls.__signature__ = cls.__signature__.replace(
        parameters=[
            arg.replace(default=Form(...))
            for arg in cls.__signature__.parameters.values()
        ]
    )
    return cls


@form_body
class Item(BaseModel):
    Temperature: float
    humidity: float
    wind_speed: float
    dew_point: float
    solar_radiation: float
    rainfall: float
    snowfall: float


@app.post('/test', response_model=Item)
def endpoint(item: Item = Depends(Item)):
    return item




tc = TestClient(app)

r = tc.post('/test', data={"Temperature": "Temperature",
                           "humidity": "humidity",
                           "wind_speed": "wind_speed",
                           "dew_point": "dew_point",
                           "solar_radiation": "solar_radiation",
                           "rainfall": "rainfall",
                           "snowfall": "snowfall"
                           })

# info = form_body()

@app.post("/predict")
def predict():
    # load model
    load_model = tf.keras.models.load_model('my_model2.h5')
    prediction = load_model.predict(endpoint())
    values = int(np.argmax(prediction))
    return (values)
