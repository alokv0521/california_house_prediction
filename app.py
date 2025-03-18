import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)   
# create a Flask app it acts as a initial point of the application
# model=pickle.load(open('regmodel.pkl','rb'))# Load the model from the file)
with open("regmodel.pkl", "rb") as f:
    model = pickle.load(f)

# scalar=pickle.load(open('scaling.pkl','rb'))# Load the scaled input from the file)
with open("scaling.pkl", "rb") as f:
    scalar = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_api', methods=['POST'])

def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_Data=scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output=model.predict(new_Data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict', methods=['POST'])

def predict():
    data=[float(x) for x in request.form.values()]
    final_input=scalar.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=model.predict(final_input)[0]
    return render_template("index.html",prediction_text="The House price prediction is {}".format(output))

if __name__ == '__main__':
    app.run(debug=True) 
