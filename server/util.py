import pickle
import json
import numpy as np

__locations = None
__data_columns = None
__model = None

def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __data_columns
    global __locations

    with open("./artefactos/columns.json", "r") as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[89:]

    global __model
    if __model is None:
        with open('./artefactos/home_peru_prices_model.pickle', 'rb') as f:
            __model = pickle.load(f)
    print("loading saved artifacts...done")

def get_estimated_price(location,bhk,bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = bath
    x[1] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return round(__model.predict([x])[0],2)

def get_location_names():
    return __locations

def get_data_columns():
    return __data_columns

if __name__ == '__main__':
    load_saved_artifacts()
    print(get_location_names())
    print(get_estimated_price('Pachacamac', 3, 3))
    print(get_estimated_price('Miraflores', 2, 2))
    print(get_estimated_price('SanJuanDeLurigancho',  2, 2))
    print(get_estimated_price('SantiagoDeSurco', 2, 2))

