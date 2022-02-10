import numpy as np
import pandas as pd
import flask
from flask import Flask, request, render_template, abort
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import pickle
import random

app = Flask(__name__)
limiter = Limiter(app, key_func=get_remote_address)

xgb_filename = 'MLModel/SavedModel/model_xgb_10-02-2022.sav'
xgb_model = pickle.load(open(xgb_filename, 'rb'))

kmeans_filename = 'MLModel/SavedModel/model_kmeans_10-02-2022.sav'
kmeans_model = pickle.load(open(kmeans_filename, 'rb'))


def process_input(port, ip):
    port = int(port)
    ip_arr = ip.split('.')
    df = pd.DataFrame({'dt': [port],
                       'ip1': [int(ip_arr[0])],
                       'ip2': [int(ip_arr[1])],
                       'ip3': [int(ip_arr[2])],
                       'ip4': [int(ip_arr[3])]})
    return df


@app.route('/')
@limiter.limit('5 per day')
def home():
    return render_template('index.html')


@app.route('/luckynum', methods=['POST'])
@limiter.limit('5 per day')
def predict():
    client_port = request.environ.get('REMOTE_PORT')
    client_ip_address = request.remote_addr
    input_df = process_input(client_port, client_ip_address)
    xgb_prediction = xgb_model.predict(input_df)[0]

    if xgb_prediction == 0:
        num = random.randint(0, 100)
        return render_template('output.html', prediction_text='Your Lucky Number is {}'.format(num))
    else:
        if kmeans_model.predict(input_df)[0] == 0:
            num = random.randint(0, 100)
            return render_template('output.html', prediction_text='Your Lucky Number is {}'.format(num))
        else:
            abort(403)


if __name__ == "__main__":
    app.run(debug=True)