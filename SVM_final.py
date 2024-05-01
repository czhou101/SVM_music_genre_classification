import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import utils
import os
import copy


train_split = 5
test_split = 5

# read training features
print("reading features")
train_name = 'features_split{}_all_nmfcc20.npy'
X_train = np.load(train_name.format(train_split))


# Read training labels
print("reading labels")
df = pd.read_csv('./train.csv')
y_train = df['Genre'].tolist()
y_train = y_train[:800]
y_train = np.repeat(y_train, train_split)

print("reading test features")
test_name = 'features_test_split{}_all_nmfcc20.npy'
X_test = np.load(test_name.format(test_split))


# Remove the feature columns with the lowest mutual information,
# from feat_sel.py
X_train = np.delete(X_train, 62, 1)
X_test = np.delete(X_test, 62, 1)

X_train = np.delete(X_train, 61, 1)
X_test = np.delete(X_test, 61, 1)

X_train = np.delete(X_train, 51, 1)
X_test = np.delete(X_test, 51, 1)

X_train = np.delete(X_train, 50, 1)
X_test = np.delete(X_test, 50, 1)

X_train = np.delete(X_train, 49, 1)
X_test = np.delete(X_test, 49, 1)

X_train = np.delete(X_train, 32, 1)
X_test = np.delete(X_test, 32, 1)

X_train = np.delete(X_train, 31, 1)
X_test = np.delete(X_test, 31, 1)

X_train = np.delete(X_train, 30, 1)
X_test = np.delete(X_test, 30, 1)

X_train = np.delete(X_train, 26, 1)
X_test = np.delete(X_test, 26, 1)


# normalize the data and encode the labels
encoder = LabelEncoder()
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
y_train = encoder.fit_transform(y_train)


# build and train svm
net = SVC(C=5, tol=1e-3, gamma=0.012, kernel='rbf', class_weight='balanced', probability=True, random_state=69)

print("\nTraining")
net.fit(X_train, y_train)
print("Done ")



# Evaluate
acc_train = net.score(X_train, y_train)
print("\nAccuracy on train = %0.4f " % acc_train)


# do majority voting between split audio to find 
# label
y_predict10 = net.predict(X_test)
y_predictions = np.array_split(y_predict10, 200)

y_predict = []
import statistics as st
for idx, prediction in enumerate(y_predictions):
    y_predict.append(st.mode(prediction))
   

# decode labels and write csv
y_predict = encoder.inverse_transform(y_predict)
utils.write_csv(y_predict=y_predict, fname='svm_split5_fs')