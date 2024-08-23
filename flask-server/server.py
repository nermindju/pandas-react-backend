from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import os
import traceback
import re
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


global_df = None
model_path = 'pandas-react-backend/flask-server/models/XGBoost_model.joblib'
XGBoost_model = joblib.load(model_path)

model_to_type = {
            'Arteon': 'Sedan', 'Passat': 'Sedan', 'Golf': 'Hatchback',
            'Scirocco': 'Sports/Coupe', 'Tiguan': 'SUV', 'Polo': 'Hatchback',
            'Golf Variant': 'Van', 'Sharan': 'Monovolume', 'Bora': 'Van',
            'Caddy': 'Kombi', 'T4': 'Kombi', 'Golf Plus': 'Monovolume',
            'Touran': 'Monovolume', 'CC': 'Sedan', 'Jetta': 'Sedan',
            'Amarok': 'Pick-up', 'Golf Alltrack': 'Van', 'T5': 'Van',
            'Touareg': 'SUV', 'T-Roc': 'SUV', 'T5 Caravelle': 'Van',
            'Golf Sportsvan': 'Monovolume', 'ID.4': 'SUV', 'ID.5': 'SUV',
            'ID.3': 'Hatchback', 'Buba / Beetle': 'Hatchback', 'T5 Multivan': 'Van',
            'Passat Alltrack': 'Van', 'Passat Variant': 'Van', 'Golf GTE': 'Hatchback',
            'T6': 'Van', 'T7 Multivan': 'Van', 'Vento': 'Sedan',
            'LT': 'Van', 'T2': 'Van', 'T4 Caravelle': 'Van', 'Phaeton': 'Sedan',
            'Up!': 'Hatchback', 'Tiguan Allspace': 'SUV', 'T-Cross': 'SUV', 'e-Golf': 'Hatchback',
            'Fox': 'Hatchback', 'Crafter': 'Van', 'T6 Caravelle': 'Van', 'Eos': 'Van',
            'New Beetle': 'Hatchback', 'Corrado': 'Sports/Coupe', 'Polo Cross': 'SUV', 'T3': 'Van',
            'Taigo': 'SUV', 'T4 Multivan': 'Van', 'Buggy': 'Off-Road', 'Santana': 'Off-Road',
            'T6 Multivan': 'Van', 'T3 Caravelle': 'Van', 'T3 Multivan': 'Van', 'Buba / Käfer / New Beetle': 'Hatchback',
            '181': 'Off-Road', 'Routan': 'Monovolume', 'Atlas': 'SUV', 'Polo Plus': 'Hatchback',
            'Polo Variant': 'Van', 'T6 Shuttle': 'Van', 'ID.7': 'SUV', 'Iltis': 'Off-Road',
            'ID.6': 'SUV', 'XL1': 'Sedan', 'T5 Shuttle': 'Van', 'T1': 'Van', 'Lupo':'Small car'
        }

vehicle_type_mapping = {
    'Sedan': 10,
    'Hatchback': 3,
    'Caravan': 1,
    'Small car': 11,
    'Monovolume': 4,
    'SUV': 9,
    'Van': 13,
    'Sports/Coupe': 12,
    'Caddy': 0,
    'Other': 7,
    'Convertible': 2,
    'Pick-up': 8,
    'Off-Road':5,
    'Oldtimer':6
}       
        
drivetrain_mapping = {'FWD':0, 'AWD':1, 'RWD':2}

fuel_mapping = {'Diesel':1, 'Petrol': 0, 'Gas': 4, 'Electro': 2, 'Hybrid': 3} 

tranmission_mapping = {'Manual':0, 'Automatic':1}

doors_mapping = {'4/5':0, '2/3':1}

default_values = {
    'displacement': 1.9,
    'kilowatts': 77,
    'mileage' : 80_000,
    'year' : 2018,
    'rimsize' : 18.0,
    'type' : 'Hatchback',
    'drivetrain' : 'FWD',
    'fuel' : 'Diesel',
    'transmission' : 'Manual',
    'doors' : '4/5',
    'cruisecontrol' : 0,
    'aircondition' : 0,
    'navigation' : 0,
    'registration' : 0,
    'parkingsensors' : 0
}

# Parking sensors {'Front':3, 'Rear':2, 'Front and Rear':1, '-':0}
# Cruise control {'True':1, 'False':0}
# Registration {'True':1, 'False':0}
# Navigation {'True':1, 'False':0}
# Air condition {'True':1, 'False':0}
# Drivetrain {'FWD':0, 'AWD':1, 'RWd'2:}
# Fuel {'Petrol':0, 'Diesel':1, 'Electro':2, 'Hybrid':3, 'Gas':4}
# Transmission {'Manual':0, 'Automatic':1}
# Doors {'4/5':0, '2/3':1}



@app.route("/upload_csv", methods=["POST"])
def upload_csv():
    global global_df
    global data_vw
    global numeric_vw_data 

    # Define the list of required columns
    required_columns = [
       'createdat', 'displacement', 'doors', 'fuel', 'kilowatts', 'location',
       'manufacturer', 'mileage', 'model', 'price', 'title', 'year', 'color',
       'cruisecontrol', 'drivetrain', 'emissionstandard', 'parkingsensors',
       'rimsize', 'transmission', 'type', 'aircondition', 'navigation',
       'registration'
    ]

    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        try:
            global_df = pd.read_csv(file_path)
            global_df = global_df.replace({np.nan: None})

            # Standardize column names by stripping and lowering case
            global_df.columns = global_df.columns.str.strip().str.lower()

            # Check if all required columns are present in the uploaded CSV
            missing_columns = [col for col in required_columns if col not in global_df.columns]
            if missing_columns:
                return jsonify({"error": f"Missing required columns: {', '.join(missing_columns)}"}), 400

            # Drop any columns that are not in the required_columns list
            global_df = global_df[required_columns]

            # Continue with the existing data processing steps
            columns_to_fill = ['aircondition', 'registration', 'navigation', 'cruisecontrol']
            global_df[columns_to_fill] = global_df[columns_to_fill].fillna('False')

            global_df.dropna(subset=['displacement', 'year', 'doors', 'kilowatts', 'mileage', 'fuel'], inplace=True)
            global_df['rimsize'].fillna(global_df['rimsize'].median(), inplace=True)
            global_df['transmission'].fillna(global_df['transmission'].mode()[0], inplace=True)
            global_df['drivetrain'] = global_df['drivetrain'].replace({'Prednji':'FWD', 'Zadnji':'RWD', 'Sva četiri':'AWD'})
            global_df['type'] = global_df['type'].replace({'Limuzina':'Sedan', 'Kombi': 'Van', 'Monovolumen':'Monovolume', 'Karavan':'Caravan', 'Malo auto':'Small car', 
                                                           'Kabriolet':'Convertible', 'Ostalo': 'Other', 'Sportski/kupe':'Sports/Coupe'})
            global_df['fuel'] = global_df['fuel'].replace({'Dizel':'Diesel', 'Benzin':'Petrol', 'Hibrid':'Hybrid', 'Elektro':'Electro', 'Plin':'Gas'})
            global_df['transmission'] = global_df['transmission'].replace({'Manuelni':'Manual', 'Automatik':'Automatic'})
            global_df['parkingsensors'] = global_df['parkingsensors'].replace({'Naprijed':'Front', 'Nazad':'Rear', 'Naprijed i nazad':'Front and Rear'})

            drivetrain_values = {
                    'Arteon': 'FWD',
                    'Passat': 'FWD',
                    'Golf': 'FWD',
                    'Scirocco': 'FWD',
                    'Tiguan': 'AWD',
                    'Polo': 'FWD',
                    'Golf Variant': 'FWD',
                    'Sharan': 'AWD',
                    'Bora': 'FWD',
                    'Caddy': 'FWD',
                    'T4 Kombi': 'RWD',
                    'Golf Plus': 'FWD',
                    'Touran': 'FWD',
                    'Passat CC': 'FWD',
                    'Jetta': 'FWD',
                    'Amarok': 'AWD',
                    'Golf Alltrack': 'AWD',
                    'T5 Kombi': 'RWD',
                    'Touareg': 'AWD',
                    'T-Roc': 'AWD',
                    'T5 Caravelle': 'RWD',
                    'Golf Sportsvan': 'FWD',
                    'ID.4': 'AWD',
                    'ID.5': 'AWD',
                    'ID.3': 'FWD',
                    'Buba / Beetle': 'FWD',
                    'T5 Multivan': 'RWD',
                    'Passat Alltrack': 'AWD',
                    'Passat Variant': 'FWD',
                    'Golf GTE': 'AWD',
                    'T6 Kombi': 'AWD',
                    'T7 Multivan': 'AWD',
                    'Vento': 'FWD',
                    'LT': 'RWD',
                    'T2': 'RWD',
                    'T4 Caravelle': 'RWD',
                    'Phaeton': 'AWD',
                    'Up!': 'FWD',
                    'Tiguan Allspace': 'AWD',
                    'T4 Drugi': 'AWD',
                    'T-Cross': 'FWD',
                    'e-Golf': 'FWD',
                    'Fox': 'FWD',
                    'Crafter': 'FWD',
                    'T6 Caravelle': 'AWD',
                    'T5 Drugi': 'RWD',
                    'Eos': 'FWD',
                    'New Beetle': 'FWD',
                    'Corrado': 'FWD',
                    'Polo Cross': 'FWD',
                    'T3 Kombi': 'RWD',
                    'Taigo': 'FWD',
                    'T4 Multivan': 'RWD',
                    'Buggy': 'RWD',
                    'T3': 'RWD',
                    'Santana': 'FWD',
                    'T4': 'RWD',
                    'T5': 'RWD',
                    'T6 Multivan': 'AWD',
                    'T3 Caravelle': 'RWD',
                    'T3 Multivan': 'RWD',
                    'CC': 'FWD',
                    'Buba / Käfer / New Beetle': 'FWD',
                    'T6 Drugi': 'AWD',
                    '181': 'RWD',
                    'Routan': 'AWD',
                    'Atlas': 'AWD',
                    'Polo Plus': 'FWD',
                    'Polo Variant': 'FWD',
                    'T3 Drugi': 'RWD',
                    'T6': 'AWD',
                    'T6 Shuttle': 'AWD',
                    'ID.7': 'AWD',
                    'Iltis': 'AWD',
                    'ID.6': 'AWD',
                    'XL1': 'FWD',
                    'T5 Shuttle': 'RWD',
                    'T1': 'RWD',
                    'Lupo':'FWD'
                }

            global_df['drivetrain'] = global_df['drivetrain'].fillna(global_df['model'].map(drivetrain_values))
            global_df['emissionstandard'].fillna(global_df['emissionstandard'].mode()[0], inplace=True)
            global_df['color'].fillna('-', inplace=True)
            global_df['parkingsensors'].fillna('-', inplace=True)

            global_df.drop(global_df[(global_df['displacement'] > 7.0) | (global_df['displacement'] <= 0)].index, inplace=True)
            global_df.drop(global_df[(global_df['kilowatts'] > 600) | (global_df['kilowatts'] < 35)].index, inplace=True)
            global_df.drop(global_df[(global_df['price'] > 1_000_000) | (global_df['price'] < 500)].index, inplace=True)
            global_df.drop(global_df[(global_df['model'] == 'Passat') & (global_df['price'] > 300_000)].index, inplace=True)
            global_df.drop(global_df[(global_df['price'] > 100_000) & (global_df['year'] < 2006)].index, inplace=True)
            global_df.drop(global_df[global_df['year'] <= 0].index, inplace=True)
            global_df.drop(global_df[global_df['year'] < 1950].index, inplace=True)
            global_df.drop(global_df[(global_df['model'] == 'ID.3') & (global_df['year'] < 2020)].index, inplace=True)
            global_df.drop(global_df[(global_df['model'] == 'ID.4') & (global_df['year'] < 2020)].index, inplace=True)
            global_df.drop(global_df[(global_df['model'] == 'ID.5') & (global_df['year'] < 2020)].index, inplace=True)
            global_df['type'].replace('Terenac','SUV', inplace=True)
            global_df['type'].replace('Pick up', 'Pick-up', inplace=True)
            global_df['type'].replace('Off Road', 'Off-Road', inplace=True)


            global_df['mileage'] = global_df['mileage'].astype('float')
            global_df.drop(global_df[global_df['mileage'] > 1_000_000].index, inplace=True)

            global_df.drop(global_df[(global_df['type'] == 'Hatchback') & (global_df['price'] > 200_000)].index, inplace=True)

            global_df.drop(global_df[(global_df['year'] < 2021) & (global_df['mileage'] < 1000)].index, inplace=True)

            # Filter for Volkswagen data
            data_vw = global_df[global_df['manufacturer'] == 'Volkswagen']
            data_vw['model'] = data_vw['model'].apply(standardize_model)
            data_vw = data_vw[data_vw['model'].notna()]
            data_vw.loc[(data_vw['model'] == 'Golf') & (data_vw['type'] == 'Sedan'), 'type'] = 'Hatchback'
            data_vw.loc[(data_vw['model'] == 'Golf') & (data_vw['type'] == 'SUV'), 'type'] = 'Hatchback'
            data_vw.loc[(data_vw['model'] == 'Golf') & (data_vw['type'] == 'Monovolume'), 'type'] = 'Caravan'
            data_vw.loc[(data_vw['model'] == 'Golf') & (data_vw['type'] == 'Off-Road'), 'type'] = 'Hatchback'
            data_vw.loc[(data_vw['model'] == 'Golf') & (data_vw['type'] == 'Other'), 'type'] = 'Hatchback'
            data_vw.loc[(data_vw['model'] == 'Golf') & (data_vw['type'] == 'Small car'), 'type'] = 'Hatchback'
            data_vw.loc[(data_vw['model'] == 'Golf') & (data_vw['type'] == 'Sports/Coupe'), 'type'] = 'Hatchback'
            data_vw.loc[(data_vw['model'] == 'Arteon') & (data_vw['type'] == 'SUV'), 'type'] = 'Sedan'
            data_vw.loc[(data_vw['model'] == 'ID.3') & (data_vw['type'] == 'Sedan'), 'type'] = 'Hatchback'
            data_vw.loc[(data_vw['model'] == 'ID.3') & (data_vw['type'] == 'Small car'), 'type'] = 'Hatchback'
            data_vw.loc[(data_vw['model'] == 'ID.4') & (data_vw['type'] == 'Sedan'), 'type'] = 'SUV'
            data_vw.loc[(data_vw['model'] == 'Taigo') & (data_vw['type'] == 'Sedan'), 'type'] = 'SUV'
            data_vw.loc[(data_vw['model'] == 'Tiguan') & (data_vw['type'] == 'Sedan'), 'type'] = 'SUV'
            data_vw.loc[(data_vw['model'] == 'CC') & (data_vw['type'] == 'SUV'), 'type'] = 'Sports/Coupe'
            data_vw.loc[(data_vw['model'] == 'CC') & (data_vw['type'] == 'Van'), 'type'] = 'Sports/Coupe'
            data_vw.loc[(data_vw['model'] == 'CC') & (data_vw['type'] == 'Other'), 'type'] = 'Sports/Coupe'
            data_vw.loc[(data_vw['model'] == 'CC') & (data_vw['type'] == 'Sedan'), 'type'] = 'Sports/Coupe'
            data_vw.loc[(data_vw['model'] == 'CC') & (data_vw['type'] == 'Small car'), 'type'] = 'Sports/Coupe'
            data_vw.loc[(data_vw['model'] == 'Caddy') & (data_vw['type'] == 'Small car'), 'type'] = 'Caddy'
            data_vw.loc[(data_vw['model'] == 'Caddy') & (data_vw['type'] == 'Monovolume'), 'type'] = 'Caddy'
            data_vw.loc[(data_vw['model'] == 'Caddy') & (data_vw['type'] == 'Sedan'), 'type'] = 'Caddy'
            data_vw.loc[(data_vw['model'] == 'Sharan') & (data_vw['type'] == 'Sedan'), 'type'] = 'Monovolume'
            data_vw.loc[(data_vw['model'] == 'Sharan') & (data_vw['type'] == 'SUV'), 'type'] = 'Monovolume'
            data_vw.loc[(data_vw['model'] == 'Sharan') & (data_vw['type'] == 'Caravan'), 'type'] = 'Monovolume'
            data_vw.loc[(data_vw['model'] == 'Sharan') & (data_vw['type'] == 'Caddy'), 'type'] = 'Monovolume'
            data_vw.loc[(data_vw['model'] == 'Sharan') & (data_vw['type'] == 'Other'), 'type'] = 'Monovolume'
            data_vw.loc[(data_vw['model'] == 'Sharan') & (data_vw['type'] == 'Small car'), 'type'] = 'Monovolume'
            data_vw.loc[(data_vw['model'] == 'Sharan') & (data_vw['type'] == 'Small car'), 'type'] = 'Monovolume'
            data_vw.loc[(data_vw['model'] == 'Sharan') & (data_vw['type'] == 'Van'), 'type'] = 'Monovolume'
            data_vw.loc[(data_vw['model'] == 'Passat') & (data_vw['type'] == 'SUV'), 'type'] = 'Sedan'
            data_vw.loc[(data_vw['model'] == 'Polo') & (data_vw['type'] == 'SUV'), 'type'] = 'Hatchback'
            data_vw.loc[(data_vw['model'] == 'Touareg') & (data_vw['type'] == 'Sedan'), 'type'] = 'SUV'
            data_vw.loc[(data_vw['model'] == 'Touran') & (data_vw['type'] == 'SUV'), 'type'] = 'Monovolume'
            data_vw.loc[(data_vw['model'] == 'Touran') & (data_vw['type'] == 'Sedan'), 'type'] = 'Monovolume'
            data_vw.loc[(data_vw['model'] == 'Touran') & (data_vw['type'] == 'Caravan'), 'type'] = 'Monovolume'
            data_vw.loc[(data_vw['model'] == 'Touran') & (data_vw['type'] == 'Convertible'), 'type'] = 'Monovolume'
            data_vw.loc[(data_vw['model'] == 'Touran') & (data_vw['type'] == 'Other'), 'type'] = 'Monovolume'
            data_vw.loc[(data_vw['model'] == 'Touran') & (data_vw['type'] == 'Small car'), 'type'] = 'Monovolume'
            data_vw.loc[(data_vw['model'] == 'Golf Plus') & (data_vw['type'] == 'SUV'), 'type'] = 'Monovolume'
            data_vw.loc[(data_vw['model'] == 'Golf Plus') & (data_vw['type'] == 'Sedan'), 'type'] = 'Monovolume'
            data_vw.loc[(data_vw['model'] == 'Golf Plus') & (data_vw['type'] == 'Caravan'), 'type'] = 'Monovolume'
            data_vw.loc[(data_vw['model'] == 'Golf Plus') & (data_vw['type'] == 'Other'), 'type'] = 'Monovolume'
            data_vw.loc[(data_vw['model'] == 'Golf Plus') & (data_vw['type'] == 'Small car'), 'type'] = 'Monovolume'
            data_vw.loc[(data_vw['model'] == 'Golf Plus') & (data_vw['type'] == 'Van'), 'type'] = 'Monovolume'  
            data_vw.loc[(data_vw['model'] == 'Tiguan') & (data_vw['type'] == 'Monovolume'), 'type'] = 'SUV'
            data_vw.loc[(data_vw['model'] == 'Tiguan') & (data_vw['type'] == 'Caravan'), 'type'] = 'SUV'
            data_vw.loc[(data_vw['model'] == 'Tiguan') & (data_vw['type'] == 'Off-Road'), 'type'] = 'SUV'
            data_vw.loc[(data_vw['model'] == 'Tiguan') & (data_vw['type'] == 'Off-Road'), 'type'] = 'SUV'
            data_vw.loc[(data_vw['model'] == 'Tiguan') & (data_vw['type'] == 'Other'), 'type'] = 'SUV'
            data_vw.loc[(data_vw['model'] == 'Tiguan') & (data_vw['type'] == 'Small car'), 'type'] = 'SUV'
            data_vw.loc[(data_vw['model'] == 'Tiguan') & (data_vw['type'] == 'Sports/Coupe'), 'type'] = 'SUV'
            data_vw.loc[(data_vw['model'] == 'Tiguan') & (data_vw['type'] == 'Van'), 'type'] = 'SUV'
            data_vw.loc[(data_vw['model'] == 'Touareg') & (data_vw['type'] == 'Monovolume'), 'type'] = 'SUV'
            data_vw.loc[(data_vw['model'] == 'Touareg') & (data_vw['type'] == 'Caravan'), 'type'] = 'SUV'
            data_vw.loc[(data_vw['model'] == 'Touareg') & (data_vw['type'] == 'Small car'), 'type'] = 'SUV'
            data_vw.loc[(data_vw['model'] == 'Touareg') & (data_vw['type'] == 'Sports/Coupe'), 'type'] = 'SUV'
            data_vw.loc[(data_vw['model'] == 'Passat') & (data_vw['type'] == 'Monovolume'), 'type'] = 'Caravan'
            data_vw.loc[(data_vw['model'] == 'Amarok') & (data_vw['type'] == 'Other'), 'type'] = 'Pick-up'
            data_vw.loc[(data_vw['model'] == 'Amarok') & (data_vw['type'] == 'SUV'), 'type'] = 'Pick-up'
            data_vw.loc[(data_vw['model'] == 'Amarok') & (data_vw['type'] == 'Sports/Coupe'), 'type'] = 'Pick-up'
            data_vw.loc[(data_vw['model'] == 'Arteon') & (data_vw['type'] == 'Sports/Coupe'), 'type'] = 'Sedan'
            data_vw.loc[(data_vw['model'] == 'Arteon') & (data_vw['type'] == 'Small car'), 'type'] = 'Sedan'
            data_vw.loc[(data_vw['model'] == 'Bora') & (data_vw['type'] == 'Small car'), 'type'] = 'Sedan'
            data_vw.loc[(data_vw['model'] == 'Passat Alltrack') & (data_vw['type'] == 'SUV'), 'type'] = 'Off-Road'

            data_vw['drivetrain'] = data_vw['drivetrain'].fillna(data_vw['model'].map(drivetrain_values))
            data_vw['type'] = data_vw['type'].fillna(data_vw['model'].map(model_to_type))
            data_vw['type'] = data_vw['type'].replace({'Kombi':'Van'})
            data_vw.loc[(data_vw['model'] == 'CC') & (data_vw['type'] == 'Sedan'), 'type'] = 'Sports/Coupe'
            data_vw.loc[(data_vw['model'] == 'Caddy') & (data_vw['type'] == 'Van'), 'type'] = 'Caddy'
            data_vw.loc[(data_vw['model'] == 'Caddy') & (data_vw['type'] == 'Caravan'), 'type'] = 'Caddy'
            data_vw.loc[(data_vw['model'] == 'Bora') & (data_vw['type'] == 'Van'), 'type'] = 'Caravan'
            data_vw.loc[(data_vw['model'] == 'Passat Alltrack') & (data_vw['type'] == 'Van'), 'type'] = 'Caravan'
            data_vw.loc[(data_vw['model'] == 'Passat Variant') & (data_vw['type'] == 'Van'), 'type'] = 'Caravan'
            data_vw.loc[(data_vw['model'] == 'T-Roc') & (data_vw['type'] == 'Caravan'), 'type'] = 'SUV'
            data_vw.loc[(data_vw['model'] == 'T-Roc') & (data_vw['type'] == 'Monovolume'), 'type'] = 'SUV'
            data_vw.loc[(data_vw['model'] == 'T-Roc') & (data_vw['type'] == 'Sedan'), 'type'] = 'SUV'
            data_vw.loc[(data_vw['model'] == 'T5') & (data_vw['type'] == 'Sports/Coupe'), 'type'] = 'Van'
            data_vw.loc[(data_vw['model'] == 'T5') & (data_vw['type'] == 'Convertible'), 'type'] = 'Van'
            data_vw.loc[(data_vw['model'] == 'T5') & (data_vw['type'] == 'Caravan'), 'type'] = 'Van'
            data_vw.loc[(data_vw['model'] == 'T5') & (data_vw['type'] == 'Caravan'), 'type'] = 'Van'
            data_vw.loc[(data_vw['model'] == 'T5') & (data_vw['type'] == 'SUV'), 'type'] = 'Van'
            data_vw.loc[(data_vw['model'] == 'Up!') & (data_vw['type'] == 'Hatchback'), 'type'] = 'Small car'
            data_vw.loc[(data_vw['model'] == 'Up!') & (data_vw['type'] == 'Sedan'), 'type'] = 'Small car'
            data_vw.loc[(data_vw['model'] == 'Golf Sportsvan') & (data_vw['type'] == 'Caravan'), 'type'] = 'Monovolume'
            data_vw.loc[(data_vw['model'] == 'Golf Sportsvan') & (data_vw['type'] == 'Sedan'), 'type'] = 'Monovolume'
            data_vw.loc[(data_vw['model'] == 'Golf Sportsvan') & (data_vw['type'] == 'Small car'), 'type'] = 'Monovolume'
            data_vw.loc[(data_vw['model'] == 'Golf Sportsvan') & (data_vw['type'] == 'Other'), 'type'] = 'Monovolume'
            data_vw.loc[(data_vw['model'] == 'Golf Variant') & (data_vw['type'] == 'Small car'), 'type'] = 'Caravan'
            data_vw.loc[(data_vw['model'] == 'Golf Alltrack') & (data_vw['type'] == 'Van'), 'type'] = 'Caravan'
            data_vw.loc[(data_vw['model'] == 'Lupo') & (data_vw['type'] == 'Sedan'), 'type'] = 'Small car'
            data_vw.loc[(data_vw['model'] == 'T5 Caravelle') & (data_vw['type'] == 'SUV'), 'type'] = 'Van'
            data_vw.loc[(data_vw['model'] == 'Taigo') & (data_vw['type'] == 'Caravan'), 'type'] = 'SUV'
            data_vw.loc[(data_vw['model'] == 'Taigo') & (data_vw['type'] == 'Small car'), 'type'] = 'SUV'
            data_vw.loc[(data_vw['model'] == 'Vento') & (data_vw['type'] == 'Small car'), 'type'] = 'Sedan'
            data_vw.loc[(data_vw['model'] == '181') & (data_vw['type'] == 'Caravan'), 'type'] = 'Off-Road'
            data_vw.loc[(data_vw['model'] == 'Crafter') & (data_vw['type'] == 'Caravan'), 'type'] = 'Van'
            data_vw.loc[(data_vw['model'] == 'Eos') & (data_vw['type'] == 'Van'), 'type'] = 'Convertible'
            data_vw.loc[(data_vw['model'] == 'Eos') & (data_vw['type'] == 'Small car'), 'type'] = 'Convertible'
            data_vw.loc[(data_vw['model'] == 'Golf GTE') & (data_vw['type'] == 'Sedan'), 'type'] = 'Hatchback'
            data_vw.loc[(data_vw['model'] == 'Golf GTE') & (data_vw['type'] == 'Small car'), 'type'] = 'Hatchback'
            data_vw.loc[(data_vw['model'] == 'Polo Cross') & (data_vw['type'] == 'Small car'), 'type'] = 'SUV'
            data_vw.loc[(data_vw['model'] == 'Polo Cross') & (data_vw['type'] == 'Sedan'), 'type'] = 'SUV'
            data_vw.loc[(data_vw['model'] == 'T-Cross') & (data_vw['type'] == 'Small car'), 'type'] = 'SUV'
            data_vw.loc[(data_vw['model'] == 'Buggy') & (data_vw['type'] == 'Sedan'), 'type'] = 'Off-Road'
            data_vw.loc[(data_vw['model'] == 'Polo Variant') & (data_vw['type'] == 'Van'), 'type'] = 'Caravan'
            data_vw.loc[(data_vw['model'] == 'Routan') & (data_vw['type'] == 'Small car'), 'type'] = 'Van'
            data_vw.loc[(data_vw['model'] == 'Tiguan Allspace') & (data_vw['type'] == 'Caravan'), 'type'] = 'SUV'
            data_vw.loc[(data_vw['model'] == 'Tiguan Allspace') & (data_vw['type'] == 'Monovolume'), 'type'] = 'SUV'
            data_vw.loc[(data_vw['model'] == 'e-Golf') & (data_vw['type'] == 'Small car'), 'type'] = 'Hatchback'
            data_vw.loc[(data_vw['model'] == 'e-Golf') & (data_vw['type'] == 'Smalll car'), 'type'] = 'Hatchback'
            data_vw.loc[(data_vw['model'] == 'e-Golf') & (data_vw['type'] == 'Caravan'), 'type'] = 'Hatchback'


            data_vw.drop(data_vw[(data_vw['model'] == 'Golf') & (data_vw['type'] == 'Van')].index, inplace=True)
            data_vw.drop(data_vw[(data_vw['model'] == 'Passat') & (data_vw['type'] == 'Convertible')].index, inplace=True)
            data_vw.drop(data_vw[(data_vw['model'] == 'Passat') & (data_vw['type'] == 'Other')].index, inplace=True)
            data_vw.drop(data_vw[(data_vw['model'] == 'Passat') & (data_vw['type'] == 'Small car')].index, inplace=True)
            data_vw.drop(data_vw[(data_vw['model'] == 'Passat') & (data_vw['type'] == 'Van')].index, inplace=True)
            data_vw.drop(data_vw[(data_vw['model'] == 'Passat') & (data_vw['type'] == 'Sports/Coupe')].index, inplace=True)
            data_vw.drop(data_vw[(data_vw['model'] == 'Polo') & (data_vw['type'] == 'Other')].index, inplace=True)
            data_vw.drop(data_vw[(data_vw['model'] == 'Bora') & (data_vw['type'] == 'Other')].index, inplace=True)
            data_vw.drop(data_vw[(data_vw['model'] == 'Jetta') & (data_vw['type'] == 'Small car')].index, inplace=True)
            data_vw.drop(data_vw[(data_vw['model'] == 'T-Roc') & (data_vw['type'] == 'Other')].index, inplace=True)
            data_vw.drop(data_vw[(data_vw['model'] == 'T4') & (data_vw['type'] == 'Other')].index, inplace=True)
            data_vw.drop(data_vw[(data_vw['model'] == 'T5') & (data_vw['type'] == 'Other')].index, inplace=True)
            data_vw.drop(data_vw[(data_vw['model'] == 'Tiguan') & (data_vw['type'] == 'Oldtimer')].index, inplace=True)
            data_vw.drop(data_vw[(data_vw['model'] == 'Touran') & (data_vw['type'] == 'Oldtimer')].index, inplace=True)
            data_vw.drop(data_vw[(data_vw['model'] == 'Fox') & (data_vw['type'] == 'Other')].index, inplace=True)
            data_vw.drop(data_vw[(data_vw['model'] == 'T2') & (data_vw['type'] == 'Other')].index, inplace=True)
            data_vw.drop(data_vw[(data_vw['model'] == 'LT') & (data_vw['type'] == 'Other')].index, inplace=True)
            data_vw.drop(data_vw[(data_vw['model'] == 'Buba / Käfer / New Beetle') & (data_vw['type'] == 'Sedan')].index, inplace=True)
            data_vw.drop(data_vw[(data_vw['model'] == 'Buggy') & (data_vw['type'] == 'Other')].index, inplace=True)
            data_vw.drop(data_vw[(data_vw['model'] == 'T6') & (data_vw['type'] == 'Other')].index, inplace=True)
            print(data_vw[(data_vw['model'] == 'T6') & (data_vw['type'] == 'Other')])
            print(f'Nullovi {data_vw.isnull().sum()}')
            # nan_drivetrain_models = data_vw[data_vw['drivetrain'].isna()]['model']

            # Display the models with NaN values in 'drivetrain'
            # print("Models with NaN values in drivetrain column:")
            # print(nan_drivetrain_models.unique())

            # Check if data_vw is empty 
            if data_vw.empty:
                return jsonify({"error": "Filtered Volkswagen data is empty after processing."}), 400

            # Ensure no NaNs in final JSON response
            data_vw = data_vw.replace({np.nan: None})
            numeric_vw_data = data_vw[['price', 'mileage', 'displacement', 'kilowatts', 'year']]
            numeric_vw_data = numeric_vw_data.replace({np.nan: None})

            # Save the Volkswagen data as a CSV file
            vw_csv_filename = 'volkswagen_data.csv'
            vw_csv_path = os.path.join(app.config['UPLOAD_FOLDER'], vw_csv_filename)
            data_vw.to_csv(vw_csv_path, index=False)


            # Convert data to JSON-serializable format
            data = data_vw.to_dict(orient="records")
            description = global_df.describe().round(2).replace({np.nan: None}).to_dict()

            return jsonify({
                "data": data,
                "shape": global_df.shape,
                "description": description
            })

        except Exception as e:
            print(f"Error: {str(e)}")
            print(traceback.format_exc())
            return jsonify({"error": str(e)}), 500

def standardize_model(model):
    if pd.isna(model) or model == 'Drugi':
        return None
    model = re.sub(r'\s*\(.*\)', '', model)
    model = model.replace('Passat CC', 'CC')
    model = model.replace('Drugi', '').strip()
    model = model.replace('Kombi', '').strip()
    return model


@app.route("/get_columns", methods=['GET'])
def get_columns():
    if data_vw is None:
        return jsonify({"error": "No data available. Please upload a CSV file first."}), 400
    
    columns = data_vw.columns.tolist()
    return jsonify({"columns": columns})

@app.route("/get_XGBoost_prediction", methods=['POST'])
def get_prediction():

    volkswagen_data = pd.read_csv('uploads/volkswagen_data.csv')
    try:
        # Get JSON data from the request
        data = request.json

        # Function to get value with fallback to default if value is missing or empty
        def get_value(key):
            return data.get(key) if data.get(key) not in [None, ""] else default_values[key]

        # Use the get_value function to handle missing/empty values
        vehicle_type_str = get_value('type')
        vehicle_drivetrain_str = get_value('drivetrain')
        fuel_mapping_str = get_value('fuel')
        transmission_mapping_str = get_value('transmission')
        doors_mapping_str = get_value('doors')

        numeric_vehicle_type = vehicle_type_mapping.get(vehicle_type_str)
        if numeric_vehicle_type is None:
            return jsonify({"error": f"Unknown vehicle type: {vehicle_type_str}"}), 400
        numeric_vehicle_drivetrain = drivetrain_mapping.get(vehicle_drivetrain_str)
        if numeric_vehicle_drivetrain is None:
            return jsonify({"error": f"Unknown drivetrain type: {vehicle_drivetrain_str}"}), 400
        numeric_vehicle_fuel = fuel_mapping.get(fuel_mapping_str)
        if numeric_vehicle_fuel is None:
            return jsonify({"error": f"Unknown fuel type: {fuel_mapping_str}"}), 400
        numeric_vehicle_transmission = tranmission_mapping.get(transmission_mapping_str)
        if numeric_vehicle_transmission is None:
            return jsonify({"error": f"Unknown transmission type: {transmission_mapping_str}"}), 400
        numeric_vehicle_doors = doors_mapping.get(doors_mapping_str)
        if numeric_vehicle_doors is None:
            return jsonify({"error": f"Unknown doors type: {doors_mapping_str}"}), 400

        # Prepare features with default values for missing/empty data
        features = [
            get_value('displacement'),
            get_value('kilowatts'),
            get_value('mileage'),
            get_value('year'),
            get_value('rimsize'),
            numeric_vehicle_drivetrain,
            numeric_vehicle_doors,
            numeric_vehicle_type,
            get_value('cruisecontrol'),
            get_value('aircondition'),
            get_value('navigation'),
            get_value('registration'),
            numeric_vehicle_fuel,
            get_value('parkingsensors'),
            numeric_vehicle_transmission
        ]

        features = [features]  # Convert to 2D array (list of lists)
        prediction = XGBoost_model.predict(features)[0]

        price_range_low = prediction - 5000 
        price_range_high = prediction + 5000 
        kw_low = get_value('kilowatts') - 30
        kw_high = get_value('kilowatts') + 30
        displacement_low = get_value('displacement') - 0.4
        displacement_high = get_value('displacement') + 0.4
        year_low = get_value('year') - 2
        year_high = get_value('year') + 2


        # Filter data_vw DataFrame based on these criteria
        filtered_vehicles = volkswagen_data[
            (volkswagen_data['price'] >= price_range_low) &
            (volkswagen_data['price'] <= price_range_high) &
            (volkswagen_data['kilowatts'] >= kw_low) &
            (volkswagen_data['kilowatts'] <= kw_high) &
            (volkswagen_data['displacement'] >= displacement_low) &
            (volkswagen_data['displacement'] <= displacement_high) &
            (volkswagen_data['year'] >= year_low) &
            (volkswagen_data['year'] <= year_high) & 
            (volkswagen_data['type'] == vehicle_type_str)
        ]

        # Select top 10 vehicles (or adjust the number as needed)
        filtered_vehicles = filtered_vehicles.head(10)

        # Convert the filtered vehicles to JSON-serializable format
        filtered_vehicles_json = filtered_vehicles.to_dict(orient='records')

        # Return the prediction and the filtered vehicles
        return jsonify({
            "prediction": str(prediction),
            "vehicles": filtered_vehicles_json
        })

    except KeyError as e:
        return jsonify({"error": f"Missing parameter: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/hist_plot", methods=["GET"])
def get_hist_plot_data():

    volkswagen_data = pd.read_csv('uploads/volkswagen_data.csv')
    try:
        if volkswagen_data is None:
            return jsonify({"error": "No data available. Please upload a CSV file first."}), 400
        
        # Define the price ranges
        bins = [0, 5000, 15000, 30000, 50000, 70000, float('inf')]
        labels = ['$0-$5000', '$5000-$15000', '$15000-$30000', '$30000-$50000', '$50000-$70000', '$70000+']

        # Categorize the prices into the defined ranges
        volkswagen_data['Price Range'] = pd.cut(volkswagen_data['price'], bins=bins, labels=labels, right=False)

        # Count the number of vehicles in each price range
        range_counts = volkswagen_data['Price Range'].value_counts().sort_index()

        # Convert to the desired format
        data = [{"priceRange": price_range, "count": count} for price_range, count in range_counts.items()]

        return jsonify(data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/model_ranking", methods=["GET"])
def get_model_ranking_data():
    volkswagen_data = pd.read_csv('uploads/volkswagen_data.csv')

    try:
        if volkswagen_data is None or 'model' not in volkswagen_data.columns:
            return jsonify({"error": "No data available or 'model' column is missing."}), 400

        # Count the occurrences of each car model
        model_counts = volkswagen_data['model'].value_counts().head(5)

        # Format the data as required
        top_models = [{"id": model, "value": count} for model, count in model_counts.items()]

        return jsonify(top_models)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/models_average_price", methods=['GET'])
def get_models_average_price():
    volkswagen_data = pd.read_csv('uploads/volkswagen_data.csv')

    try:
        if volkswagen_data is None or 'model' not in volkswagen_data.columns:
            return jsonify({"error": "No data available or 'model' column is missing"}), 400
        
        top_25_models = volkswagen_data['model'].value_counts().head(25).index.tolist()
        
        filtered_df = volkswagen_data[volkswagen_data['model'].isin(top_25_models)]
        
        average_prices = filtered_df.groupby('model')['price'].mean().round(2).reset_index()
        average_prices_list = average_prices.to_dict(orient='records')

        return jsonify(average_prices_list)

    except Exception as e:
        return jsonify({"error": str(e)}), 500    
    
    # top 25 modela
    

@app.route("/get_models_price_box", methods=["GET"])
def get_models_price_box_data():
    volkswagen_data = pd.read_csv('uploads/volkswagen_data.csv')



    try:
        if volkswagen_data is None:
            return jsonify({"error": "No data available. Please upload a CSV file first."}), 400
        
        # Get the top 10 models by value counts
        top_10_models = volkswagen_data['model'].value_counts().head(10).index.tolist()

        # Define the vehicle types to include
        selected_types = ['SUV', 'Hatchback', 'Sedan', 'Caravan']

        # Filter the dataset based on the selected types and top 10 models
        filtered_data = volkswagen_data[
            (volkswagen_data['type'].isin(selected_types)) & 
            (volkswagen_data['model'].isin(top_10_models))
        ]

        # Group the data and calculate price statistics
        grouped_data = filtered_data.groupby(['model', 'type'])['price'].agg([
            ('min_price', 'min'),
            ('25_percentile', lambda x: np.percentile(x, 25)),
            ('median_price', 'median'),
            ('75_percentile', lambda x: np.percentile(x, 75)),
            ('max_price', 'max')
        ]).reset_index()

        # Format the data for the response
        formatted_data = []
        for _, row in grouped_data.iterrows():
            model = row['model']
            v_type = row['type']
            prices = [
                row['min_price'],
                row['25_percentile'],
                row['median_price'],
                row['75_percentile'],
                row['max_price']
            ]

            formatted_data.append({
                "group": model,
                "data": [{"key": v_type, "value": prices}]
            })

        return jsonify(formatted_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    
    # top 10 modela


@app.route("/type_minmax_price", methods=['GET'])
def get_type_minmax_price():
    volkswagen_data = pd.read_csv('uploads/volkswagen_data.csv')
    try:
        if volkswagen_data is None or 'type' not in volkswagen_data.columns:
            return jsonify({"error": "No data available or 'type' column is missing"}), 400
        
        # Group data by 'type' and calculate min and max prices
        grouped_data = volkswagen_data.groupby(['type'])['price'].agg([
            ('minPrice', 'min'),
            ('maxPrice', 'max')
        ]).reset_index()
        
        # Format the data to match the desired structure
        formatted_data = [
            {
                'priceRange': row['type'],
                'minPrice': row['minPrice'],
                'maxPrice': row['maxPrice']
            }
            for _, row in grouped_data.iterrows()
        ]

        return jsonify(formatted_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/model_listings", methods=['GET'])
def model_listings():

    volkswagen_data = pd.read_csv('uploads/volkswagen_data.csv')
    try:
        if volkswagen_data is None or 'type' not in volkswagen_data.columns or 'model' not in volkswagen_data.columns:
            return jsonify({"error": "No data available or 'type'/'model' column is missing"}), 400

        modelCounts = volkswagen_data['model'].value_counts().to_dict()

        response_data = {
            'model_to_type': model_to_type,
            'modelCounts': modelCounts
        }

        return jsonify(response_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    
@app.route("/get_line_plot_data", methods=['GET'])
def get_line_plot_data():
    volkswagen_data = pd.read_csv('uploads/volkswagen_data.csv')
    
    try:
        if volkswagen_data is None or 'model' not in volkswagen_data.columns or 'price' not in volkswagen_data.columns:
            return jsonify({"error": "No data available or 'model'/'price' column is missing"}), 400
        
        # Get the top 5 models by count
        top_models = volkswagen_data['model'].value_counts().nlargest(5).index.tolist()

        # Filter data for only these top models
        filtered_data = volkswagen_data[volkswagen_data['model'].isin(top_models)]
        
        # Group data by model and calculate min and median prices
        grouped_data = filtered_data.groupby('model')['price'].agg([
            ('25PercentilePrice', lambda x: x.quantile(0.25)),
            ('medianPrice', 'median')
        ]).reset_index()

        price_data = [
            {
                'id': 'Price Range',
                'data': [{'x': row['model'], 'y': row['25PercentilePrice']} for _, row in grouped_data.iterrows()]
            },
            {
                'id': 'Median Price',
                'data': [{'x': row['model'], 'y': row['medianPrice']} for _, row in grouped_data.iterrows()]
            }
        ]

        return jsonify(price_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/get_prices', methods=['GET'])
def get_prices():
    volkswagen_data = pd.read_csv('uploads/volkswagen_data.csv')

    try:
        if volkswagen_data is None or 'price' not in volkswagen_data.columns:
            return jsonify({"error": "No data available or 'price' column is missing"}), 400
    
        avg_price = volkswagen_data['price'].mean().round(2)
        median_price = volkswagen_data['price'].median().round(2)
        quantile_price_25 = volkswagen_data['price'].quantile(0.25).round(2)
        quantile_price_75 = volkswagen_data['price'].quantile(0.75).round(2)

        return jsonify({
            'mean_price': avg_price,
            'median_price': median_price,
            '25_percent_price' : quantile_price_25,
            '75_percent_price' : quantile_price_75
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# @app.route("/group_by", methods=["POST"])
# def group_by():
#     global global_df

#     if global_df is None:
#         return jsonify({"error": "No data available. Please upload a CSV file first."}), 400

#     request_data = request.get_json()
#     column_name = request_data.get("column_name")
#     value_column = request_data.get("value_column")
#     aggregation_function = request_data.get("aggregation_function")

#     if column_name is None or column_name not in global_df.columns:
#         return jsonify({"error": "Invalid column name"}), 400

#     if value_column is None or value_column not in global_df.columns:
#         return jsonify({"error": "Invalid value column"}), 400

#     if aggregation_function not in ["mean", "sum", "count", "max", "min"]:
#         return jsonify({"error": "Invalid aggregation function"}), 400

#     try:
#         grouped_df = global_df.groupby(column_name).agg({value_column: aggregation_function}).reset_index().round(2)
#         grouped_data = grouped_df.to_dict(orient="records")
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

#     return jsonify({
#         "grouped_data": grouped_data
#     })

# @app.route("/fillna", methods=["POST"])
# def fillna():
#     global global_df

#     if global_df is None:
#         return jsonify({"error": "No data available. Please upload a CSV file first."}), 400

#     request_data = request.get_json()
#     fill_value = request_data.get("fill_value")
#     method = request_data.get("method")
#     column_name = request_data.get("column_name")

#     if fill_value is None and method is None:
#         return jsonify({"error": "Either fill_value or method is required"}), 400

#     if fill_value is not None and method is not None:
#         return jsonify({"error": "Cannot specify both fill_value and method"}), 400

#     if column_name and column_name not in global_df.columns:
#         return jsonify({"error": f"Invalid column name: {column_name}"}), 400

#     try:
#         if column_name:
#             if fill_value is not None:
#                 global_df[column_name].fillna(fill_value, inplace=True)
#             else:
#                 global_df[column_name].fillna(method=method, inplace=True)
#         else:
#             if fill_value is not None:
#                 global_df.fillna(fill_value, inplace=True)
#             else:
#                 global_df.fillna(method=method, inplace=True)
#         data = global_df.to_dict(orient="records")
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

#     return jsonify({
#         "data": data,
#         "shape": global_df.shape
#     })

# @app.route("/dropna", methods=["POST"])
# def dropna():
#     global global_df

#     if global_df is None:
#         return jsonify({"error": "No data available. Please upload a CSV file first."}), 400

#     request_data = request.get_json()
#     axis = request_data.get("axis", 0)  # 0 for rows, 1 for columns
#     how = request_data.get("how", "any")  # 'any' or 'all'

#     if axis not in [0, 1]:
#         return jsonify({"error": "Invalid axis value"}), 400

#     if how not in ["any", "all"]:
#         return jsonify({"error": "Invalid how value"}), 400

#     try:
#         global_df.dropna(axis=axis, how=how, inplace=True)
#         data = global_df.to_dict(orient="records")
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

#     return jsonify({
#         "data": data,
#         "shape": global_df.shape
#     })

# @app.route("/count_plot", methods=['POST'])
# def count_plot():
#     if data_vw is None:
#         return jsonify({"error": "No data available. Please upload a CSV file first."}), 400
    
#     data = request.json
#     x_column = data.get('x_column')
    
#     if x_column not in data_vw.columns:
#         return jsonify({"error": f"Column '{x_column}' not found in the data."}), 400

#     plt.figure(figsize=(7, 4))
#     sns.countplot(x=x_column, data=data_vw)
#     plt.title(f"Count Plot for {x_column}")

#     img = BytesIO()
#     plt.savefig(img, format='png')
#     img.seek(0)
#     img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
#     plt.close()

#     return jsonify({"count_plot": img_base64})

# @app.route("/scatter_plot", methods=['POST'])
# def scatter_plot():
#     if data_vw is None:
#         return jsonify({"error": "No data available. Please upload a CSV file first."}), 400
    
#     data = request.json
#     x_column = data.get('x_column')
#     y_column = data.get('y_column')
    
#     print(f"Received x_column: {x_column}, y_column: {y_column}")  # Debugging line

#     if x_column not in data_vw.columns:
#         return jsonify({"error": f"Column '{x_column}' not found in the data."}), 400

#     if y_column and y_column not in data_vw.columns:
#         return jsonify({"error": f"Column '{y_column}' not found in the data."}), 400

#     plt.figure(figsize=(7, 4))
#     sns.scatterplot(x=x_column, y=y_column, data=data_vw)
#     plt.title(f"Scatter Plot for {x_column} vs {y_column}")

#     img = BytesIO()
#     plt.savefig(img, format='png')
#     img.seek(0)
#     img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
#     plt.close()

#     return jsonify({"scatter_plot": img_base64})


# @app.route("/heat_map", methods=['POST'])
# def heat_map():
#     if data_vw is None:
#         return jsonify({"error": "No data available. Please upload a CSV file first."}), 400

#     plt.figure(figsize=(7, 4))
#     sns.heatmap(data=numeric_vw_data.corr(), annot=True, cmap='coolwarm')
#     plt.title(f"Heat map")
#     plt.xticks(rotation=45)

#     img = BytesIO()
#     plt.savefig(img, format='png')
#     img.seek(0)
#     img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
#     plt.close()

#     return jsonify({"heat_map": img_base64})

if __name__ == "__main__":  
    app.run(debug=True)