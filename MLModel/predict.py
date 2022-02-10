import pickle
import pandas as pd

xgb_filename = 'SavedModel/model_xgb_10-02-2022.sav'
xgb_model = pickle.load(open(xgb_filename, 'rb'))
kmeans_filename = 'SavedModel/model_kmeans_10-02-2022.sav'
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


port_num = '11425'
ip_address = '10.0.0.1'

input_df = process_input(port_num, ip_address)
xgb_prediction = xgb_model.predict(input_df)[0]

if xgb_prediction == 0:
    print("XGB: IP is alright.")
else:
    if kmeans_model.predict(input_df)[0] == 0:
        print("KMeans: IP is alright.")
    else:
        print("IP is suspicious.")

