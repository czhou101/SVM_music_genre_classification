import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split



train_split = 5
test_split = 5

# Reading training data
print("reading features")
train_name = 'features_split{}_all_nmfcc20.npy'
X = np.load(train_name.format(train_split))

# reading training labels
print("reading labels")
df = pd.read_csv('./train.csv')
y = df['Genre'].tolist()
y = y[:800]
y = np.repeat(y, train_split)


# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


# normalize and encode the labels
encoder = LabelEncoder()
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
y_train = encoder.fit_transform(y_train)
y_test = encoder.fit_transform(y_test)


# build and evaluate SVM
net = SVC(C=5, tol=1e-3, gamma=0.012, kernel='rbf', class_weight='balanced', probability=True, random_state=69)

print("\nTraining")
net.fit(X_train, y_train)
print("Done ")



# Evaluate
acc_train = net.score(X_train, y_train)
print("\nAccuracy on train = %0.4f " % acc_train)
acc_test = net.score(X_test, y_test)
print("Accuracy on test = %0.4f " % acc_test)

