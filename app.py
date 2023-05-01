import pickle
from flask import Flask, render_template, request,app,jsonify,url_for
import numpy as np
import pandas as pd


app = Flask(__name__) #flask app instance

# model = pickle.load(open('classmodel.pkl','rb'))
model = pd.read_pickle(open('classmodel.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods =['POST']) #create an api where we can send a post request from postman or any other tool

def predict_api():

    
    data = request.json['data'] #whenever hit the api, the input will be fetched here
    # print(data)
    
    df = pd.DataFrame(data)
    print(df)
    # encoder = pickle.load(open('encoder.pkl','rb'))
    encoder= pd.read_pickle(open('encoder.pkl','rb'))
    data_point_encoded = encoder.transform(df)
    data_x = data_point_encoded.values[0].reshape(1,-1)
    scaler = pd.read_pickle(open('scaler.pkl','rb'))
    data_x_scaled = scaler.transform(data_x[:,1:5])
    data_x_scaled = np.concatenate((data_x_scaled, data_x[:, 5:]), axis=1)
    output = model.predict(data_x_scaled)
    print(output[0])
    return jsonify(int(output[0])) #numpy integer cannot be jsonify directly.

@app.route('/submit_form',methods = ['POST'])
def submit_form():
    
    try:
        data = dict(request.form)
        print(data)
    except:
        print('Data is not loaded')
    for key,value in data.items():
        data[key] = [value]
    
    df= pd.DataFrame(data)
    encoder= pd.read_pickle(open('encoder.pkl','rb'))
    data_point_encoded = encoder.transform(df)
    data_x = data_point_encoded.values[0].reshape(1,-1)
    scaler = pd.read_pickle(open('scaler.pkl','rb'))
    data_x_scaled = scaler.transform(data_x[:,1:5])
    data_x_scaled = np.concatenate((data_x_scaled, data_x[:, 5:]), axis=1)
    output = model.predict(data_x_scaled)[0]
    if output == 0:
        return render_template('home.html',prediction_text = "The applicant has low credit risk")
    else:
        return render_template('home.html',prediction_text = "The applicant has high credit risk")

    
    

if __name__ == "__main__":
    app.run(debug =True)