import pandas as pd
from preprocessing import processor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from xgboost import XGBClassifier
import time
import pickle


df = pd.read_csv('Dataset/dataset_sdn.csv')
df = processor(df)
features = df[['dt', 'ip1', 'ip2', 'ip3', 'ip4']]
cls = df['label']

X_train, X_val, Y_train, Y_val = train_test_split(features, cls, test_size=0.15, shuffle=True)

s1 = time.time()
xgb = XGBClassifier()
xgb.fit(X_train, Y_train)
y_predict = xgb.predict(X_val)
predictions = [round(value) for value in y_predict]
accuracy = accuracy_score(Y_val, predictions)
print("Accuracy of XGB model over the given dataset is: %.2f%%" % (accuracy * 100.0))
e1 = time.time()
print("Time taken by XGB: {:.2f} seconds \n".format(e1 - s1))
filename = 'SavedModel/model_xgb_10-02-2022.sav'
pickle.dump(xgb, open(filename, 'wb'))

s1 = time.time()
kmeans = KMeans(n_clusters=2)
df['kmeans_prediction'] = kmeans.fit_predict(features)

count = 0
num_rows = df.shape[0]
for i in range(0, num_rows):
    if df.iloc[i]['kmeans_prediction'] == df.iloc[i]['label']:
        count = count + 1
accuracy = count / num_rows

print("Accuracy of KMeans model over the given dataset is: %.2f%%" % (accuracy * 100.0))
e1 = time.time()
print("Time taken by KMeans: {:.2f} seconds \n".format(e1 - s1))
filename = 'SavedModel/model_kmeans_10-02-2022.sav'
pickle.dump(kmeans, open(filename, 'wb'))
