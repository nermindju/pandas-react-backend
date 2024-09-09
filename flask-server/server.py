from flask import Flask, jsonify, request, Response
from flask_cors import CORS
import shutil
from datetime import datetime
import pandas as pd
import os
import fnmatch
import glob
import traceback
import re
import numpy as np
import joblib
import math
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib
from openai import OpenAI
matplotlib.use('Agg')

pd.set_option("display.max_columns", None)

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
global_df = None
global_data = pd.read_csv('uploads/volkswagen_data.csv')
global_model_choice = None
selected_brand = 'volkswagen'
volkswagen_data=pd.read_csv('uploads/volkswagen_data.csv')
default_model_path = f'{selected_brand}_models/'
default_model_prefix = 'xgboost_model'
for file_name in os.listdir(default_model_path):
    if fnmatch.fnmatch(file_name, f'{default_model_prefix}*.joblib'):
        global_model_choice = os.path.join(default_model_path, file_name)
        break


import logging
logging.basicConfig(level=logging.DEBUG)

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

@app.route('/get_ai_message', methods=['POST'])
def get_ai_message():
    try:
        data = request.json
        message = data.get('message')
        messages = data.get('messages', [])

        if not message:
            print("No message provided")
            return jsonify({"error": "Message is required"}), 400

        ai_response = ""
        stream = client.chat.completions.create(
            model="ft:gpt-4o-mini-2024-07-18:personal::A2GTlhVH",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "system", "content": f"Relevant data: {volkswagen_data}"},
                {"role": "system", "content": "The bot is a factual chatbot that provides pandas DataFrame queries."},
                {"role": "user", "content": message}
            ],
            stream=True
        )

        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                ai_response += chunk.choices[0].delta.content

        query = ai_response.split('Query: ')[-1].strip()
        variable_name, expression = query.split('=', 1)
        expression = expression.strip().replace('df', 'volkswagen_data')

        try:
            query_result = eval(expression)
            print("query result: ", query_result)
            if query_result is None:
                raise ValueError("Query execution failed or no data found.")
        except Exception as e:
            print(f"Eval Error: {e}")
            query_result = None

        if query_result is None:
            query_result_str = "No data available."
        else:
            query_result_str = str(query_result)
        
        gpt4o_mini_response = client.chat.completions.create(
          model="gpt-4o-mini",
          messages=[
              {"role": "system", "content": "You are a helpful assistant."}, 
              {"role": "system", "content": f"The user has retrieved data from a database about Volkswagen cars. Based on the provided data: {query_result_str}"},
              {"role": "system", "content": "Your task is to answer the user's question accurately using only this data if it is relevant to the question."},
              {"role": "system", "content": "When Using the provided data, mention that that answer is about cars on the page that the user is using."},
              {"role": "user", "content": message},
              *messages
          ],
          stream=True
        )

        final_response = ""
        for chunk in gpt4o_mini_response:
            if chunk.choices[0].delta.content is not None:
                final_response += chunk.choices[0].delta.content

        return jsonify({"ai_message": final_response})

    except Exception as e:
        print(f"Server Error: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

    
@app.route("/get_csv", methods=["GET"])
def get_csv():
    global_data = pd.read_csv(f'uploads/{selected_brand}_data.csv')
    try:
        if global_data is not None:
            logging.info("DataFrame loaded successfully.")
            logging.info("DataFrame shape: %s", global_data.shape)
            
            data = global_data.to_dict(orient="records")
            return jsonify(data)
        else:
            logging.error("DataFrame is None.")
            return jsonify({"error": "DataFrame is None"}), 400
    except Exception as e:
        logging.error("Error occurred: %s", traceback.format_exc())
        return jsonify({"error": str(e)}), 500

model_to_type = {
            'Arteon': 'Sedan', 'Passat': 'Sedan', 'Golf': 'Hatchback',
            'Scirocco': 'Sports/Coupe', 'Tiguan': 'SUV', 'Polo': 'Hatchback',
            'Golf Variant': 'Van', 'Sharan': 'Monovolume', 'Bora': 'Van',
            'Caddy': 'Caddy', 'T4': 'Van', 'Golf Plus': 'Monovolume',
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
    'Sedan': 9,
    'Hatchback': 3,
    'Caravan': 1,
    'Small car': 10,
    'Monovolume': 4,
    'SUV': 8,
    'Van': 12,
    'Sports/Coupe': 11,
    'Caddy': 0,
    'Convertible': 2,
    'Pick-up': 7,
    'Off-Road':5,
    'Oldtimer':6
}       
audi_model_to_type = {
    'A6': 'Sedan',
    'A3': 'Hatchback',
    '100': 'Sedan',
    'A4': 'Caravan',
    'A2': 'Small car',
    '80': 'Sedan',
    'A6 Allroad': 'Caravan',
    'A8': 'Sedan',
    'A1': 'Small car',
    'TT': 'Sports/Coupe',
    'Q7': 'SUV',
    'A5': 'Sedan',
    'Q5': 'SUV',
    'A4 Allroad': 'Caravan',
    'S3': 'Sports/Coupe',
    'Q3': 'SUV',
    'A7': 'Sedan',
    'SQ5': 'SUV',
    'Q8': 'SUV',
    'SQ7': 'SUV',
    'S6': 'Sedan',
    'RSQ8': 'SUV',
    'SQ8': 'SUV',
    'RSQ3': 'SUV',
    'RS7': 'Sedan',
    'RS6': 'Caravan',
    'RS5': 'Sedan',
    'Q2': 'SUV',
    'S8': 'Sedan',
    'S4': 'Sedan',
    'RS3': 'Sports/Coupe',
    'SQ2': 'SUV',
    'S5': 'Sedan',
    'e-tron GT': 'Sedan',
    '90': 'Sedan',
    'S1': 'Small car',
    'e-tron': 'SUV',
    'S7': 'Sedan',
    'RS4': 'Caravan',
    'R8': 'Sports/Coupe',
    'V8': 'Sedan'
}
        
drivetrain_mapping = {'FWD':1, 'AWD':0, 'RWD':2}

fuel_mapping = {'Diesel':0, 'Petrol': 4, 'Gas': 2, 'Electro': 1, 'Hybrid': 3} 

tranmission_mapping = {'Manual':0, 'Automatic':1}

doors_mapping = {'4/5':1, '2/3':0}

sensors_mapping = {'Front':1, 'Rear':3, 'Front and Rear':2, '-':0}

default_values = {
    'displacement': 1.9,
    'kilowatts': 77,
    'mileage' : 80_000,
    'year' : 2011,
    'rimsize' : 18.0,
    'drivetrain' : 'FWD',
    'doors' : '4/5',
    'type' : 'Hatchback',
    'cruisecontrol' : 0,
    'aircondition' : 0,
    'navigation' : 0,
    'registration' : 0,
    'fuel' : 'Diesel',
    'parkingsensors' : 0,
    'transmission' : 'Manual'
}   

# Parking sensors {'Front':1, 'Rear':3, 'Front and Rear':2, '-':0}
# Cruise control {'True':1, 'False':0}
# Registration {'True':1, 'False':0}
# Navigation {'True':1, 'False':0}
# Air condition {'True':1, 'False':0}
# Drivetrain {'FWD':1, 'AWD':0, 'RWd'2:}
# Fuel {'Petrol':4, 'Diesel':0, 'Electro':1, 'Hybrid':3, 'Gas':2}
# Transmission {'Manual':0, 'Automatic':1}
# Doors {'4/5':1, '2/3':0}



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
            global_df['transmission'] = global_df['transmission'].replace({'Manuelni':'Manual', 'Automatik':'Automatic', 'Polu-automatik':'Semi-automatic', 'Poluautomatik':'Semi-automatic'})
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
            global_df['uvoz'] = global_df['title'].str.contains('stranac|uvoz', case=False, regex=True).astype(int)
            global_df['udaren'] = global_df['title'].str.contains('havarisan|udaren', case=False, regex=True).astype(int)


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
            data_vw.drop(data_vw[(data_vw['model'] == 'Golf') & (data_vw['price'] > 400_000)].index, inplace=True)
            data_vw.drop(data_vw[(data_vw['model'] == 'Touran') & (data_vw['year'] < 2003)].index, inplace=True)
            print(f'Nullovi {data_vw.isnull().sum()}')

            audi_data = global_df[global_df['manufacturer'] == 'Audi']
            audi_data.drop(audi_data[audi_data['model'] == 'QUATTRO'].index, inplace=True)
            audi_data.drop(audi_data[audi_data['model'] == '307'].index, inplace=True)
            audi_data.drop(audi_data[audi_data['model'] == 'Drugi'].index, inplace=True)
            def map_other_types(data, model_to_type):
                # Identify all rows with "Other" in the Type column
                other_rows = data['type'] == 'Other'
    
                data.loc[other_rows, 'type'] = data.loc[other_rows, 'model'].map(model_to_type).fillna('Other')
    
                return data

            # Apply the mapping function to the DataFrame
            audi_data = map_other_types(audi_data, audi_model_to_type)
            print(audi_data[audi_data['type'] == 'Other'])
            audi_data.drop(audi_data[audi_data['model'] == 'Cabriolet'].index, inplace=True)
            audi_data.loc[(audi_data['model'] == 'A3') & (audi_data['year'] < 2012), 'type'] = 'Hatchback'  
            audi_data.loc[(audi_data['model'] == 'A3') & (audi_data['year'] >= 2012) & (audi_data['type'] == 'Small car'), 'type'] = 'Hatchback'  
            print(audi_data[(audi_data['model'] == 'A3') & (audi_data['year'] >= 2012) & (audi_data['type'] == 'Small car')])
            audi_data['type'] = audi_data['type'].fillna(audi_data['model'].map(audi_model_to_type))
            def fill_drivetrain_mode(df):
                # Calculate mode for each model
                mode_drivetrain = df.groupby('model')['drivetrain'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)

                # Fill NaN in drivetrain with the mode of corresponding model
                df['drivetrain'] = df.apply(
                    lambda row: mode_drivetrain[row['model']] if pd.isna(row['drivetrain']) else row['drivetrain'],
                    axis=1)

                return df

            audi_data = fill_drivetrain_mode(audi_data)
            audi_data.dropna(inplace=True)
            audi_data.drop(audi_data[audi_data['drivetrain'] == 'RWD'].index, inplace=True)
            audi_data.drop(audi_data[(audi_data['model'] == '80') & (audi_data['price'] > 15_000)].index, inplace=True)
            models_audi = ['A6', 'A3', '100', 'A4', 'A2', '80', 'A6 Allroad', 'A8', 'A1',
                    'TT', 'Q7', 'A5', 'Q5', 'A4 Allroad', 'S3', 'Q3', 'A7', 'SQ5',
                    'Q8', 'SQ7', 'S6', 'RSQ8', 'SQ8', 'RSQ3', 'RS7', 'RS6', 'RS5',
                    'Q2', 'Q4 e-tron', 'S8', 'S4', 'RS3', 'SQ2', 'S5', 'e-tron GT',
                    '90', 'S1', 'e-tron', 'S7', 'RS4', 'R8', 'V8', 'Q8 e-tron']

            def filter_df_by_model_in_title(df, known_models):
                # Convert known models to lowercase and create full search patterns like "Audi A3", "Audi A4", etc.
                search_patterns = [f'audi {model.lower()}' for model in known_models]

                # Filter rows where the title contains any of the patterns
                filtered_df = df[df['title'].str.lower().apply(lambda title: any(pattern in title for pattern in search_patterns))]

                # Return the filtered DataFrame
                return filtered_df

            # Apply the function to filter the DataFrame
            filtered_audi_df = filter_df_by_model_in_title(audi_data, models_audi)
            print(filtered_audi_df.shape)
            filtered_audi_df.drop(filtered_audi_df[filtered_audi_df['model'] == 'Q8 e-tron'].index, inplace=True)
            filtered_audi_df.loc[(filtered_audi_df['model'] == 'Q4 e-tron'), 'model'] = 'e-tron'
            filtered_audi_df.loc[(filtered_audi_df['model'] == 'A6') & (filtered_audi_df['type'] == 'Van'), 'type'] = 'Sedan'
            filtered_audi_df.loc[(filtered_audi_df['model'] == 'A4 Allroad'), 'type'] = 'Caravan'
            filtered_audi_df.loc[(filtered_audi_df['model'] == 'A4'), 'type'] = 'Sedan'
            filtered_audi_df.loc[(filtered_audi_df['model'] == 'A6 Allroad'), 'type'] = 'Caravan'
            filtered_audi_df.loc[(filtered_audi_df['model'] == 'Q3'), 'type'] = 'SUV'
            filtered_audi_df.loc[(filtered_audi_df['model'] == 'Q5'), 'type'] = 'SUV'
            filtered_audi_df.loc[(filtered_audi_df['model'] == 'Q7'), 'type'] = 'SUV'
            filtered_audi_df.loc[(filtered_audi_df['model'] == 'Q8'), 'type'] = 'SUV'
            filtered_audi_df.loc[(filtered_audi_df['model'] == 'A1'), 'type'] = 'Small car'
            filtered_audi_df.loc[(filtered_audi_df['model'] == 'A2'), 'type'] = 'Small car'
            filtered_audi_df.loc[(filtered_audi_df['model'] == 'A7'), 'type'] = 'Sedan'
            filtered_audi_df.loc[(filtered_audi_df['model'] == 'RSQ3'), 'type'] = 'SUV'
            filtered_audi_df.loc[(filtered_audi_df['model'] == 'S7'), 'type'] = 'Sedan'
            filtered_audi_df.loc[(filtered_audi_df['model'] == 'SQ5'), 'type'] = 'SUV'
            filtered_audi_df.loc[(filtered_audi_df['model'] == 'S4') & (filtered_audi_df['type'] == 'Van'), 'type'] = 'Sedan'
            filtered_audi_df.loc[(filtered_audi_df['model'] == 'A3') & (filtered_audi_df['type'] == 'Van'), 'type'] = 'Hatchback'
            filtered_audi_df.loc[(filtered_audi_df['model'] == 'S3') & (filtered_audi_df['type'] == 'Small car'), 'type'] = 'Sedan'
            filtered_audi_df.loc[(filtered_audi_df['model'] == 'e-tron') & (filtered_audi_df['type'] == 'Sedan'), 'type'] = 'SUV'
            filtered_audi_df.drop(filtered_audi_df[(filtered_audi_df['model'] == '80') & (filtered_audi_df['type'] == 'Small car')].index, inplace=True)
            filtered_audi_df.drop(filtered_audi_df[(filtered_audi_df['model'] == '80') & (filtered_audi_df['type'] == 'Monovolume')].index, inplace=True)

            a3_types_to_drop = ['SUV', 'Caravan', 'Monovolume']
            a6_types_to_drop = ['SUV', 'Convertible', 'Off-Road', 'Sports/Coupe', 'Small car']
            tt_types_to_map = ['SUV', 'Sedan', 'Small car']

            filtered_audi_df.loc[(filtered_audi_df['model'] == 'TT') & (filtered_audi_df['type'].isin(tt_types_to_map)), 'type'] = 'Sports/Coupe'
            filtered_audi_df.drop(filtered_audi_df[(filtered_audi_df['model'] == '80') & (filtered_audi_df['type'] == 'Monovolume')].index, inplace=True)
            filtered_audi_df.drop(filtered_audi_df[(filtered_audi_df['model'] == 'A3') & (filtered_audi_df['type'].isin(a3_types_to_drop))].index, inplace=True)
            filtered_audi_df.drop(filtered_audi_df[(filtered_audi_df['model'] == 'A5') & (filtered_audi_df['type'] == 'Small car')].index, inplace=True)
            filtered_audi_df.drop(filtered_audi_df[(filtered_audi_df['model'] == 'S5') & (filtered_audi_df['type'] == 'Small car')].index, inplace=True)
            filtered_audi_df.drop(filtered_audi_df[(filtered_audi_df['model'] == 'A6') & (filtered_audi_df['type'].isin(a6_types_to_drop))].index, inplace=True)

            def check_model_types(df):
                # Group by 'model' and find unique 'type' values for each model
                model_types = df.groupby('model')['type'].unique().reset_index()

                # Rename the column for clarity
                model_types.columns = ['model', 'types_present']

                # Return the DataFrame showing the models and their associated types
                return model_types

            # Apply the function to check types for each model
            model_types_df = check_model_types(filtered_audi_df)

            print(model_types_df)
            # print(filtered_audi_df[filtered_audi_df['model'] == '80']['type'].value_counts())
            # print(filtered_audi_df.shape)
            # print(filtered_audi_df.isnull().sum())
            filtered_audi_df.loc[(filtered_audi_df['fuel'] == 'Electro') & (filtered_audi_df['transmission'] == 'Manual'), 'transmission'] = 'Automatic'
            # print(filtered_audi_df[filtered_audi_df['fuel'] == 'Electro']['transmission'].value_counts())
            print(filtered_audi_df.shape)

            # Check if data_vw and audi_data is empty 
            if data_vw.empty:
                return jsonify({"error": "Filtered Volkswagen data is empty after processing."}), 400
            elif audi_data.empty:
                return jsonify({"error": "Filtered Audi data is empty after processing."}), 400
            
            

            # Ensure no NaNs in final JSON response
            data_vw = data_vw.replace({np.nan: None})
            numeric_vw_data = data_vw[['price', 'mileage', 'displacement', 'kilowatts', 'year']]
            numeric_vw_data = numeric_vw_data.replace({np.nan: None})



            # Save the Volkswagen data as a CSV file
            vw_csv = 'volkswagen_data.csv'
            vw_csv_path = os.path.join(app.config['UPLOAD_FOLDER'], vw_csv)
            data_vw.to_csv(vw_csv_path, index=False)

            # Save Audi data as a CSV file
            audi_csv = 'audi_data.csv'
            audi_csv_path = os.path.join(app.config['UPLOAD_FOLDER'], audi_csv)
            filtered_audi_df.to_csv(audi_csv_path, index=False)


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

def load_default_model():
    global global_model_choice, selected_brand
    if selected_brand:  # Ensure the brand is set
        # Dynamically set the model path based on selected_brand
        default_model_path = f'{selected_brand}_models/'
        default_model_prefix = 'xgboost_model'
        # Find the default xgboost model file in the selected brand's folder
        for file_name in os.listdir(default_model_path):
            if fnmatch.fnmatch(file_name, f'{default_model_prefix}*.joblib'):
                global_model_choice = os.path.join(default_model_path, file_name)
                break

def load_data(brand):
    global global_data, selected_brand
    selected_brand = brand
    file_name = f'uploads/{brand}_data.csv'
    global_data = pd.read_csv(file_name)
    return global_data


def train_model(model_choice):
    # Load datasets for Audi and Volkswagen
    audi_data = pd.read_csv('uploads/audi_data.csv')
    vw_data = pd.read_csv('uploads/volkswagen_data.csv')
    
    data_dict = {
        'audi': audi_data,
        'volkswagen': vw_data
    }

    results = {}
    
    for brand, data in data_dict.items():
        if data.empty:
            print(f"No data available for {brand}")
            continue

        # Preparing data for models
        encoder = LabelEncoder()
        model_data = data.copy()
        if brand == 'audi':
            model_data['type_encoded'] = model_data['type'].map({'Sedan':9, 'Hatchback':3, 'SUV':8, 'Caravan':1,
                                               'Sports/Coupe':11, 'Small car':10, 'Convertible':2,
                                               'Oldtimer':6})
        else:
            model_data['type_encoded'] = encoder.fit_transform(model_data['type'])

        # Encoding categorical features
        model_data['cruisecontrol'] = model_data['cruisecontrol'].astype('str')
        model_data['cruisecontrol_encoded'] = encoder.fit_transform(model_data['cruisecontrol'])
        model_data['aircondition'] = model_data['aircondition'].astype('str')
        model_data['aircondition_encoded'] = encoder.fit_transform(model_data['aircondition'])
        model_data['navigation'] = model_data['navigation'].astype('str')
        model_data['navigation_encoded'] = encoder.fit_transform(model_data['navigation'])
        model_data['registration'] = model_data['registration'].astype('str')
        model_data['registration_encoded'] = encoder.fit_transform(model_data['registration'])
        model_data['parkingsensors'].replace('naprijed/nazad', 'Front and Rear', inplace=True)
        model_data['parkingsensors'].replace(['Nema', 'nema'], '-', inplace=True)
        model_data['parkingsensors_encoded'] = encoder.fit_transform(model_data['parkingsensors'])
        model_data['transmission'] = model_data['transmission'].map({'Manual': 0, 'Automatic': 1, 'Semi-automatic': 1})
        model_data['fuel_encoded'] = encoder.fit_transform(model_data['fuel'])
        model_data['drivetrain_encoded'] = encoder.fit_transform(model_data['drivetrain'])
        model_data['doors_encoded'] = encoder.fit_transform(model_data['doors'])

        model_data['displacement'] = model_data['displacement'].astype('float')
        model_data['kilowatts'] = model_data['kilowatts'].astype('int')
        model_data['year'] = model_data['year'].astype('int')

        strings_to_drop = ['havarisan', 'udaren', 'stranac', 'za dijelova', 'dijelove', 'uvoz']

        def drop_rows_with_strings_in_title(df, strings_to_drop):
            if 'title' not in df.columns:
                raise ValueError("'title' column not found in the DataFrame.")
            
            # Convert the 'title' column to lowercase for case-insensitive comparison
            temp_title_lower = df['title'].str.lower()

            # Combine the strings to drop into a single regex pattern
            pattern = '|'.join(strings_to_drop)
            
            # Filter out rows where 'title' contains any of the strings
            filtered_df = df[~temp_title_lower.str.contains(pattern, case=False, na=False)]
            
            return filtered_df

        model_data = drop_rows_with_strings_in_title(model_data, strings_to_drop)
        
        X = model_data[['displacement', 'kilowatts', 'mileage', 'year', 'rimsize', 'drivetrain_encoded', 
                        'doors_encoded', 'type_encoded', 'cruisecontrol_encoded', 'aircondition_encoded', 
                        'navigation_encoded', 'registration_encoded', 'fuel_encoded', 'parkingsensors_encoded', 
                        'transmission']]
        Y = model_data['price']
        
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

        # Train the chosen model
        if model_choice == 'xgboost':
            model = XGBRegressor(gamma=0, max_depth=None, min_child_weight=5, n_estimators=100)
        elif model_choice == 'random_forest':
            model = RandomForestRegressor(max_depth=20, min_samples_leaf=1, min_samples_split=5, n_estimators=300)
        else:
            print(f"Invalid model choice for {brand}")
            return False  # Return False for invalid model choice
        
        model.fit(X_train, y_train)

        y_prediction = model.predict(X_test)

        # Calculate metrics
        r2 = round(r2_score(y_test, y_prediction), 2)
        rmse = round(math.sqrt(mean_squared_error(y_test, y_prediction)), 2)
        mae = round(mean_absolute_error(y_test, y_prediction), 2)

        # Store the results for the current brand
        results[brand] = {
            'R2 Score': r2,
            'RMSE': rmse,
            'MAE': mae
        }

        # Create the model folder if it doesn't exist
        os.makedirs(f'{brand}_models', exist_ok=True)
        
        # Define paths for current model and backup folder
        model_filename_prefix = f'{model_choice}_model'
        model_path_pattern = f'{brand}_models/{model_filename_prefix}_*.joblib'
        backup_folder = f'old_{brand}_models'
        
        # Create backup folder if it doesn't exist
        os.makedirs(backup_folder, exist_ok=True)

        # Check for existing models that match the pattern (ignore timestamp)
        existing_models = glob.glob(model_path_pattern)

        # If any matching models are found, move them to the backup folder
        for existing_model in existing_models:
            creation_time = datetime.fromtimestamp(os.path.getctime(existing_model)).strftime("%Y%m%d_%H%M%S")
            backup_model_path = f'{backup_folder}/{model_choice}_model_{creation_time}.joblib'
            shutil.move(existing_model, backup_model_path)
            print(f"Moved existing model to {backup_model_path}")
        
        # Maintain only the last 3 models in the backup folder
        backup_models = sorted(os.listdir(backup_folder), key=lambda x: os.path.getctime(f'{backup_folder}/{x}'))
        if len(backup_models) > 3:
            # Remove the oldest models until only 3 are left
            while len(backup_models) > 3:
                oldest_model = backup_models.pop(0)  # Remove the oldest model (first in sorted list)
                os.remove(f'{backup_folder}/{oldest_model}')
                print(f"Deleted oldest model: {oldest_model}")

        # Save the new model with a timestamp, including the metrics
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_model_path = f'{brand}_models/{model_choice}_model_{timestamp}.joblib'
        
        # Save the model and metrics in a dictionary object
        model_object = {
            'model': model,
            'R2 Score': r2,
            'RMSE': rmse,
            'MAE': mae
        }

        joblib.dump(model_object, new_model_path)
        print(f"Saved new model with metrics at {new_model_path}")

    return results



@app.route('/train_model', methods=['POST'])
def handle_train_model():
    try:
        model_choice = request.json.get('model')
        
        if model_choice not in ['xgboost', 'random_forest']:
            return jsonify({"message": "Invalid model choice"}), 400

        success = train_model(model_choice)

        if success:
            return jsonify({"message": f"{model_choice.replace('_', ' ').capitalize()} model trained successfully",
                            'results':success}), 200
        else:
            return jsonify({"message": f"Failed to train {model_choice.replace('_', ' ').capitalize()} model"}), 500

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"message": "Internal server error"}), 500
    

@app.route('/get_models', methods=['GET'])
def get_models():
    # Get the model type from the query params (either 'new' or 'old')
    model_type = request.args.get('model_type')
    
    if model_type not in ['new', 'old']:
        return jsonify({'status': 'error', 'message': 'Invalid model type. Choose "new" or "old".'}), 400

    brand = selected_brand  # Adjust this dynamically based on frontend if needed
    if model_type == 'new':
        folder_to_search = f'{brand}_models'
    elif model_type == 'old':
        folder_to_search = f'old_{brand}_models'

    # Get all models in the selected folder
    model_pattern = f'{folder_to_search}/*.joblib'
    available_models = glob.glob(model_pattern)

    if not available_models:
        return jsonify({'status': 'error', 'message': f'No models found in the {model_type} folder'}), 404

    # Extract only the filenames (without the full path)
    model_list = [os.path.basename(model) for model in available_models]

    return jsonify({'status': 'success', 'models': model_list}), 200


@app.route("/select_brand", methods=["POST"])
def select_brand():
    brand = request.json.get('brand')
    if brand:
        try:
            load_data(brand)
            load_default_model()
            return jsonify({"message": f"Brand '{brand}' data loaded."})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "No brand selected."}), 400


@app.route('/set_selected_model', methods=['POST'])
def set_selected_model():
    global global_model_choice
    data = request.json
    
    model_name = data.get('model_name')  # Full filename of the selected model
    model_type = data.get('model_type')  # 'new' or 'old' to know the folder

    if model_type not in ['new', 'old']:
        return jsonify({'status': 'error', 'message': 'Invalid model type. Choose "new" or "old".'}), 400

    if model_type == 'new':
        folder_to_search = f'{selected_brand}_models'
    elif model_type == 'old':
        folder_to_search = f'old_{selected_brand}_models'

    # Full path to the selected model
    model_path = os.path.join(folder_to_search, model_name)

    # Debugging: Log the path to check if it exists
    print(f"Looking for model at: {model_path}")

    if not os.path.exists(model_path):
        return jsonify({'status': 'error', 'message': f'Model {model_name} not found in {model_type} folder'}), 404

    # Set the global model choice to the selected model
    global_model_choice = model_path

    return jsonify({'status': 'success', 'message': f'Model {model_name} set as active model', 'model_path': model_path}), 200




# @app.route('/set_model_choice', methods=['POST'])
# def set_model_choice():
#     global global_model_choice
#     data = request.json
    
#     model_choice = data.get('model_choice')
    
#     if model_choice in ['xgboost', 'random_forest']:
#         global_model_choice = model_choice
#         return jsonify({'status': 'success', 'message': f'Model choice set to {model_choice}'}), 200
#     else:
#         return jsonify({'status': 'error', 'message': 'Invalid model choice'}), 400


@app.route("/get_columns", methods=['GET'])
def get_columns():
    global_data = pd.read_csv(f'uploads/{selected_brand}_data.csv')
    if global_data is None:
        return jsonify({"error": "No data available. Please upload a CSV file first."}), 400
    
    columns = global_data.columns.tolist()
    return jsonify({"columns": columns})

@app.route("/get_prediction", methods=['POST'])
def get_prediction():
    global global_model_choice

    if global_model_choice is None:
        load_default_model()

    model_path = f'{global_model_choice}'

    # Load the model object which includes the model and the metrics
    model_object = joblib.load(model_path)
    model = model_object['model']  # Extract the trained model
    r2_score = model_object.get('R2 Score', 'N/A')  # Extract R² score
    rmse = model_object.get('RMSE', 'N/A')  # Extract RMSE
    mae = model_object.get('MAE', 'N/A')  # Extract MAE

    model_data = pd.read_csv(f'uploads/{selected_brand}_data.csv')
    print(global_model_choice)
    print(selected_brand)

    try:
        data = request.json

        # Extract user-provided values
        vehicle_type_str = data.get('type')
        year_value = data.get('year')

        if not vehicle_type_str or not year_value:
            return jsonify({"error": "Vehicle type and year are required"}), 400

        # Filter data by the provided type and year to calculate modes
        filtered_data = model_data[
            (model_data['type'] == vehicle_type_str) &
            (model_data['year'] == int(year_value))
        ]

        # If no data found for the given type and year, return an error
        if filtered_data.empty:
            return jsonify({"error": "No data available for the given type and year"}), 404

        # Calculate modes for the fields where defaults are required
        def calculate_mode(column):
            if column in filtered_data:
                return filtered_data[column].mode().iloc[0]
            return None
        
        # Calculate the mean for mileage
        def calculate_mean(column):
            if column in filtered_data:
                return filtered_data[column].mean()
            return None

        mode_values = {
            'displacement': calculate_mode('displacement'),
            'kilowatts': calculate_mode('kilowatts'),
            'mileage': calculate_mean('mileage'),
            'rimsize': calculate_mode('rimsize'),
            'drivetrain': calculate_mode('drivetrain'),
            'doors': calculate_mode('doors'),
            'cruisecontrol': calculate_mode('cruisecontrol'),
            'aircondition': calculate_mode('aircondition'),
            'navigation': calculate_mode('navigation'),
            'registration': calculate_mode('registration'),
            'fuel': calculate_mode('fuel'),
            'parkingsensors': calculate_mode('parkingsensors'),
            'transmission': calculate_mode('transmission')
        }

        # Helper function to get either user-provided or mode value
        def get_value(key):
            return data.get(key) if data.get(key) not in [None, ""] else mode_values[key]

        # Use the mappings to convert to numeric values
        numeric_vehicle_type = vehicle_type_mapping.get(vehicle_type_str)
        numeric_vehicle_drivetrain = drivetrain_mapping.get(get_value('drivetrain'))
        numeric_vehicle_fuel = fuel_mapping.get(get_value('fuel'))
        numeric_vehicle_transmission = tranmission_mapping.get(get_value('transmission'))
        numeric_vehicle_doors = doors_mapping.get(get_value('doors'))
        numeric_vehicle_sensors = sensors_mapping.get(get_value('parkingsensors'))

        # Validate mappings
        if any(x is None for x in [numeric_vehicle_type, numeric_vehicle_drivetrain, numeric_vehicle_fuel, numeric_vehicle_transmission, numeric_vehicle_doors, numeric_vehicle_sensors]):
            return jsonify({"error": "Invalid mapping for vehicle attributes"}), 400

        # Prepare features with numeric data
        features = [
            float(get_value('displacement')),
            float(get_value('kilowatts')),
            float(get_value('mileage')),
            float(year_value),  # Ensure year is numeric
            float(get_value('rimsize')),
            float(numeric_vehicle_drivetrain),
            float(numeric_vehicle_doors),
            float(numeric_vehicle_type),
            float(get_value('cruisecontrol')),
            float(get_value('aircondition')),
            float(get_value('navigation')),
            float(get_value('registration')),
            float(numeric_vehicle_fuel),
            float(numeric_vehicle_sensors),
            float(numeric_vehicle_transmission)
        ]

        # Convert to NumPy array with appropriate dtype
        features = np.array([features], dtype=np.float32)  # 2D array, dtype float32

        # Make the prediction
        prediction = model.predict(features)[0].round(2)

        # Apply filters only to the user-provided fields
        filtered_vehicles = model_data[
            (model_data['type'] == vehicle_type_str) & 
            (model_data['year'] >= int(year_value) - 3) &
            (model_data['year'] <= int(year_value) + 3) &
            (model_data['price'] >= prediction - 5000) &
            (model_data['price'] <= prediction + 5000)
        ]

        if 'kilowatts' in data and data['kilowatts'] not in [None, ""]:
            kw_low = get_value('kilowatts') - 30
            kw_high = get_value('kilowatts') + 30
            filtered_vehicles = filtered_vehicles[
                (filtered_vehicles['kilowatts'] >= kw_low) &
                (filtered_vehicles['kilowatts'] <= kw_high)
            ]
        
        if 'displacement' in data and data['displacement'] not in [None, ""]:
            displacement_low = get_value('displacement') - 0.4
            displacement_high = get_value('displacement') + 0.4
            filtered_vehicles = filtered_vehicles[
                (filtered_vehicles['displacement'] >= displacement_low) &
                (filtered_vehicles['displacement'] <= displacement_high)
            ]
        
        if 'mileage' in data and data['mileage'] not in [None, ""]:
            mileage_low = get_value('mileage') - 10000
            mileage_high = get_value('mileage') + 10000
            filtered_vehicles = filtered_vehicles[
                (filtered_vehicles['mileage'] >= mileage_low) &
                (filtered_vehicles['mileage'] <= mileage_high)
            ]

        # Select top 10 vehicles
        filtered_vehicles = filtered_vehicles.head(10)

        # Convert the filtered vehicles to JSON-serializable format
        filtered_vehicles_json = filtered_vehicles.to_dict(orient='records')

        return jsonify({
            "prediction": str(prediction),
            "vehicles": filtered_vehicles_json,
            "model_metrics": {
                "R2 Score": r2_score,
                "RMSE": rmse,
                "MAE": mae
            }
        })

    except KeyError as e:
        return jsonify({"error": f"Missing parameter: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route("/hist_plot", methods=["GET"])
def get_hist_plot_data():
    global_data = pd.read_csv(f'uploads/{selected_brand}_data.csv')
    try:
        if global_data is None:
            return jsonify({"error": "No data available. Please upload a CSV file first."}), 400
        
        # Define the price ranges
        bins = [0, 5000, 15000, 30000, 50000, 70000, float('inf')]
        labels = ['$0-$5000', '$5000-$15000', '$15000-$30000', '$30000-$50000', '$50000-$70000', '$70000+']

        # Categorize the prices into the defined ranges
        global_data['Price Range'] = pd.cut(global_data['price'], bins=bins, labels=labels, right=False)

        # Count the number of vehicles in each price range
        range_counts = global_data['Price Range'].value_counts().sort_index()

        # Convert to the desired format
        data = [{"priceRange": price_range, "count": count} for price_range, count in range_counts.items()]

        return jsonify(data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/model_ranking", methods=["GET"])
def get_model_ranking_data():
    global_data = pd.read_csv(f'uploads/{selected_brand}_data.csv')
    try:
        if global_data is None or 'model' not in global_data.columns:
            return jsonify({"error": "No data available or 'model' column is missing."}), 400

        # Count the occurrences of each car model
        model_counts = global_data['model'].value_counts().head(5)

        # Format the data as required
        top_models = [{"id": model, "value": count} for model, count in model_counts.items()]

        return jsonify(top_models)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/models_average_price", methods=['GET'])
def get_models_average_price():
    global_data = pd.read_csv(f'uploads/{selected_brand}_data.csv')
    try:
        if global_data is None or 'model' not in global_data.columns:
            return jsonify({"error": "No data available or 'model' column is missing"}), 400
        
        top_25_models = global_data['model'].value_counts().head(25).index.tolist()
        
        filtered_df = global_data[global_data['model'].isin(top_25_models)]
        
        average_prices = filtered_df.groupby('model')['price'].mean().round(2).reset_index()
        average_prices_list = average_prices.to_dict(orient='records')

        return jsonify(average_prices_list)

    except Exception as e:
        return jsonify({"error": str(e)}), 500    
    
    # top 25 modela
    

@app.route("/get_models_price_box", methods=["GET"])
def get_models_price_box_data():
    global_data = pd.read_csv(f'uploads/{selected_brand}_data.csv')
    try:
        if global_data is None:
            return jsonify({"error": "No data available. Please upload a CSV file first."}), 400
        
        # Get the top 10 models by value counts
        top_10_models = global_data['model'].value_counts().head(10).index.tolist()

        # Define the vehicle types to include
        selected_types = ['SUV', 'Hatchback', 'Sedan', 'Caravan']

        # Filter the dataset based on the selected types and top 10 models
        filtered_data = global_data[
            (global_data['type'].isin(selected_types)) & 
            (global_data['model'].isin(top_10_models))
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
    global_data = pd.read_csv(f'uploads/{selected_brand}_data.csv')
    try:
        if global_data is None or 'type' not in global_data.columns:
            return jsonify({"error": "No data available or 'type' column is missing"}), 400
        
        # Group data by 'type' and calculate min and max prices
        grouped_data = global_data.groupby(['type'])['price'].agg([
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
    global_data = pd.read_csv(f'uploads/{selected_brand}_data.csv')
    try:
        if global_data is None or 'type' not in global_data.columns or 'model' not in global_data.columns:
            return jsonify({"error": "No data available or 'type'/'model' column is missing"}), 400

        modelCounts = global_data['model'].value_counts().to_dict()
        if selected_brand == 'audi':
            response_data = {
                'model_to_type': audi_model_to_type,
                'modelCounts': modelCounts
            }
        else: 
            response_data = {
                'model_to_type': model_to_type,
                'modelCounts': modelCounts
            }



        return jsonify(response_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    
@app.route("/get_line_plot_data", methods=['GET'])
def get_line_plot_data():
    global_data = pd.read_csv(f'uploads/{selected_brand}_data.csv')
    try:
        if global_data is None or 'model' not in global_data.columns or 'price' not in global_data.columns:
            return jsonify({"error": "No data available or 'model'/'price' column is missing"}), 400
        
        # Get the top 5 models by count
        top_models = global_data['model'].value_counts().nlargest(5).index.tolist()

        # Filter data for only these top models
        filtered_data = global_data[global_data['model'].isin(top_models)]
        
        # Group data by model and calculate min and median prices
        grouped_data = filtered_data.groupby('model')['price'].agg([
            ('25PercentilePrice', lambda x: x.quantile(0.25)),
            ('medianPrice', 'median'),
            ('75PercentilePrice', lambda x: x.quantile(0.75)),
        ]).reset_index()

        price_data = [
            {
                'id': '25% Price',
                'data': [{'x': row['model'], 'y': row['25PercentilePrice']} for _, row in grouped_data.iterrows()]
            },
            {
                'id': 'Median Price',
                'data': [{'x': row['model'], 'y': row['medianPrice']} for _, row in grouped_data.iterrows()]
            },
            {
                'id': '75% Price',
                'data': [{'x': row['model'], 'y': row['75PercentilePrice']} for _, row in grouped_data.iterrows()]
            }
        ]

        return jsonify(price_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/get_prices', methods=['GET'])
def get_prices():
    global_data = pd.read_csv(f'uploads/{selected_brand}_data.csv')
    try:
        if global_data is None or 'price' not in global_data.columns:
            return jsonify({"error": "No data available or 'price' column is missing"}), 400
    
        avg_price = global_data['price'].mean().round(2)
        median_price = global_data['price'].median().round(2)
        quantile_price_25 = global_data['price'].quantile(0.25).round(2)
        quantile_price_75 = global_data['price'].quantile(0.75).round(2)

        return jsonify({
            'mean_price': avg_price,
            'median_price': median_price,
            'first_quantile' : quantile_price_25,
            'third_quantile' : quantile_price_75
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@app.route('/get_correlation_heatmap', methods=['GET'])
def get_correlation_heatmap():
    global_data = pd.read_csv(f'uploads/{selected_brand}_data.csv')
    try:
        encoder = LabelEncoder()
        global_data['type'] = global_data['type'].map(vehicle_type_mapping)
        global_data['transmission'] = global_data['transmission'].map({'Manual':0, 'Automatic':1, 'Semi-automatic':1})
        # data['cruisecontrol'] = encoder.fit_transform(data['cruisecontrol'])
        # data['navigation'] = encoder.fit_transform(data['navigation'])
        # data['aircondition'] = encoder.fit_transform(data['aircondition'])
        # data['registration'] = encoder.fit_transform(data['registration'])
        # data['parkingsensors'] = encoder.fit_transform(data['parkingsensors'])
        global_data['fuel'] = encoder.fit_transform(global_data['fuel'])
        global_data['drivetrain'] = encoder.fit_transform(global_data['drivetrain'])
        # data['doors'] = encoder.fit_transform(data['doors'])
        
        # Filter only numeric data for the selected column

        numeric_data = global_data.select_dtypes(include='number')
        
        # Calculate the correlation matrix
        correlation_matrix = numeric_data.corr().round(2)
        
        # Format the correlation matrix into the required format
        heatmap_data = [
            {
                'id': row,
                'data': [{'x': col, 'y': correlation_matrix.at[row, col]} for col in correlation_matrix.columns]
            }
            for row in correlation_matrix.index
        ]
        
        return jsonify(heatmap_data)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route('/get_scatterplot_data', methods=['GET'])
def get_scatterplot_data():
    global_data = pd.read_csv(f'uploads/{selected_brand}_data.csv')
    try:
        if 'price' not in global_data.columns or 'year' not in global_data.columns or 'type' not in global_data.columns:
            return jsonify({"error": "'price', 'year', or 'type' column is missing"}), 400
            
        grouped_data = global_data.groupby('type')
        
        # Prepare the scatterplot data
        scatterplot_data = []
        for vehicle_type, group in grouped_data:
            vehicle_data = {
                "id": vehicle_type,
                "data": [{"x": row['year'], "y": row['price']} for index, row in group.iterrows()]
            }
            scatterplot_data.append(vehicle_data)
        
        return jsonify(scatterplot_data)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/get_line_plot_data_prices", methods=['GET'])
def get_line_plot_data_prices():
    global_data = pd.read_csv(f'uploads/{selected_brand}_data.csv')
    try:
        if global_data is None or 'type' not in global_data.columns or 'price' not in global_data.columns:
            return jsonify({"error": "No data available or 'type'/'price' column is missing"}), 400
        
        # Get the top 5 models by count
        top_models = global_data['type'].value_counts().nlargest(5).index.tolist()

        # Filter data for only these top models
        filtered_data = global_data[global_data['type'].isin(top_models)]
        
        # Group data by model and calculate min and median prices
        grouped_data = filtered_data.groupby('type')['price'].agg([
            ('25PercentilePrice', lambda x: x.quantile(0.25)),
            ('medianPrice', 'median'),
            ('75PercentilePrice', lambda x: x.quantile(0.75)),
        ]).reset_index()

        price_data = [
            {
                'id': '25% Price',
                'data': [{'x': row['type'], 'y': row['25PercentilePrice']} for _, row in grouped_data.iterrows()]
            },
            {
                'id': 'Median Price',
                'data': [{'x': row['type'], 'y': row['medianPrice']} for _, row in grouped_data.iterrows()]
            },
            {
                'id': '75% Price',
                'data': [{'x': row['type'], 'y': row['75PercentilePrice']} for _, row in grouped_data.iterrows()]
            }
        ]

        return jsonify(price_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/get_top5_models_avg_price_data", methods=["GET"])
def get_top5_models_avg_price_data():
    global_data = pd.read_csv(f'uploads/{selected_brand}_data.csv')
    try:
        if global_data is None or 'model' not in global_data.columns:
            return jsonify({"error": "No data available or 'model' column is missing."}), 400


        # Get top 5 models
        top5_models = global_data['model'].value_counts().head(5).index
        # Average price for each model
        average_prices = global_data[global_data['model'].isin(top5_models)].groupby('model')['price'].mean().round(2).sort_values(ascending=False)
        

        # Format the data as required
        top5_models_avg_price = [{"id": model, "value": avg_price} for model, avg_price in average_prices.items()]

        return jsonify(top5_models_avg_price)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/map", methods=['GET'])
def generate_map_data():
    try:
        top_locations = pd.read_csv(f'flask-server/location_map/geocoded_{selected_brand}_locations.csv')
    except FileNotFoundError:
        return {
            "error": f"geocoded_{selected_brand}_locations.csv file not found."
        }

    # Convert the data into a format that can be easily used by the frontend
    location_data = top_locations.to_dict(orient='records')

    return {
        "locations": location_data,
        "center": [44.0, 17.8],  # Center on Bosnia and Herzegovina
        "zoom": 7
    }


@app.route("/get_top5models_barplot_data", methods=['GET'])
def get_top5models_barplot_data():
    global_data = pd.read_csv(f'uploads/{selected_brand}_data.csv')
    try:
        if global_data is None or 'model' not in global_data.columns or 'price' not in global_data.columns:
            return jsonify({"error": "No data available or 'model' column is missing."}), 400

        # Filter data for years >= 1990
        data = global_data[global_data['year'] >= 1990]

        # Create 5-year range column
        data['year_range'] = pd.cut(data['year'], bins=range(1990, 2026, 5), right=False, labels=[f"{i}-{i+4}" for i in range(1990, 2021, 5)])

        # Group by model and year range, and calculate the average price
        grouped = data.groupby(['model', 'year_range']).price.mean().round(2).reset_index()

        # Find top 5 car models based on occurrences
        top_models = data['model'].value_counts().nlargest(5).index

        # Filter the grouped data for the top 5 car models
        final_result = grouped[grouped['model'].isin(top_models)]
        final_result.dropna(inplace=True)

        # Transform into the required JSON format
        result = []
        for year_range in final_result['year_range'].unique():
            year_data = {"year_range": year_range}
            for model in top_models:
                model_data = final_result[(final_result['model'] == model) & (final_result['year_range'] == year_range)]
                if not model_data.empty:
                    year_data[model] = model_data['price'].values[0]
            result.append(year_data)

        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route("/get_top5types_barplot_data", methods=['GET'])
def get_top5types_barplot_data():
    global_data = pd.read_csv(f'uploads/{selected_brand}_data.csv')
    try:
        if global_data is None or 'type' not in global_data.columns or 'price' not in global_data.columns:
            return jsonify({"error": "No data available or 'model' column is missing."}), 400

        # Filter data for years >= 1990
        data = global_data[global_data['year'] >= 1990]

        # Create 5-year range column
        data['year_range'] = pd.cut(data['year'], bins=range(1990, 2026, 5), right=False, labels=[f"{i}-{i+4}" for i in range(1990, 2021, 5)])

        # Group by model and year range, and calculate the average price
        grouped = data.groupby(['type', 'year_range']).price.mean().round(2).reset_index()

        # Find top 5 car models based on occurrences
        top_types = data['type'].value_counts().nlargest(5).index

        # Filter the grouped data for the top 5 car models
        final_result = grouped[grouped['type'].isin(top_types)]
        final_result.dropna(inplace=True)

        # Transform into the required JSON format
        result = []
        for year_range in final_result['year_range'].unique():
            year_data = {"year_range": year_range}
            for model in top_types:
                model_data = final_result[(final_result['type'] == model) & (final_result['year_range'] == year_range)]
                if not model_data.empty:
                    year_data[model] = model_data['price'].values[0]
            result.append(year_data)

        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/group_by", methods=["POST"])
def group_by():
    global_data = pd.read_csv(f'uploads/{selected_brand}_data.csv')


    if global_data is None:
        return jsonify({"error": "No data available. Please upload a CSV file first."}), 400

    request_data = request.get_json()
    column_name = request_data.get("column_name")
    value_column = request_data.get("value_column")
    aggregation_function = request_data.get("aggregation_function")

    if column_name is None or column_name not in global_data.columns:
        return jsonify({"error": "Invalid column name"}), 400

    if value_column is None or value_column not in global_data.columns:
        return jsonify({"error": "Invalid value column"}), 400

    if aggregation_function not in ["mean", "sum", "count", "max", "min"]:
        return jsonify({"error": "Invalid aggregation function"}), 400

    try:
        grouped_df = global_data.groupby(column_name).agg({value_column: aggregation_function}).reset_index().round(2)
        grouped_data = grouped_df.to_dict(orient="records")
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "grouped_data": grouped_data
    })

if __name__ == "__main__":  
    load_default_model()  
    app.run(debug=True)
