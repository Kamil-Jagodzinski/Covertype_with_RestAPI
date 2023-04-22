from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
import models
import dataset

app = Flask(__name__)

data = dataset.load_data()
x_train, x_test, y_train, y_test = dataset.preprocess_data(data)

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'POST'
    return response

# Define the endpoint for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json() 
    res = 0
    df = pd.DataFrame( columns=dataset.prepare_columns_names() )

    df.loc[0] = [0]*len(df.columns)
    df.at[0, 'Elevation'] = float(data['Elevation']) 
    df.at[0, 'Aspect'] = float(data['Aspect']) 
    df.at[0, 'Slope'] = float(data['Slope']) 
    df.at[0, 'Horizontal_Distance_To_Hydrology'] = float(data['Horizontal_Distance_To_Hydrology']) 
    df.at[0, 'Vertical_Distance_To_Hydrology'] = float(data['Vertical_Distance_To_Hydrology']) 
    df.at[0, 'Horizontal_Distance_To_Roadways'] = float(data['Horizontal_Distance_To_Roadways']) 
    df.at[0, 'Hillshade_9am'] = float(data['Hillshade_9am']) 
    df.at[0, 'Hillshade_Noon'] = float(data['Hillshade_Noon']) 
    df.at[0, 'Hillshade_3pm'] = float(data['Hillshade_3pm']) 
    df.at[0, 'Horizontal_Distance_To_Fire_Points'] = float(data['Horizontal_Distance_To_Fire_Points'])
    df.at[0, f"Wilderness_Area{int(data['Wilderness_Area'])}"] = 1
    df.at[0, f"Soil_type{int(data['Soil_type'])}"] = 1
    
    if( data['model'] == '0' ):
        model = models.HeuristicClassifier()
        res = model._classify( df )
    elif( data['model'] == '1' ):
        model = models.load_sklern_ml_model("models/skl_model.joblib")
        res = model.predict( df )
    else:
        std_scaler = dataset.load_pca('models/std_scaler.joblib')
        pca = dataset.load_pca('models/pca.joblib')
        std_scaler.fit(x_train)
        pca.fit(x_train)
        df = std_scaler.transform(df)
        df = pca.transform(df)

        model = models.load_nn_model("models/nn_model.h5")
        res =  np.argmax( model.predict(df), axis=-1)
    response = {'pred': models.decode_answer(res)} 
    return jsonify(response) 

if __name__ == '__main__':
    app.run( debug=True, port=5000 )
