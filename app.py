from urllib import response
import numpy as np
from flask import Flask, request, jsonify, render_template, Response
import json
import pickle


# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

#Create the route for the model prediction
@flask_app.route("/", methods = ["POST"])
def predict():
    #the API recieves a JSON with the features
    r = request.json
    #transforming them into an array for the model input
    features = [np.array([r['sepal_length'],r['sepal_width'],r['petal_length'],r['petal_width']])]
    prediction = model.predict(features)
    #creating a class key and adding the prediction as his value
    r['class'] = prediction[0]
    body = json.dumps(r)

    #defining the API response to the request
    response = Response(response=body,status=200,headers={
        'Content-Type': 'application/json'
    })

    return response

if __name__ == "__main__":
    flask_app.run(debug=True)


    