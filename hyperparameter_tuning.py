import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


train_split = 5
test_split = 5

# read training features
print("reading features")
train_name = 'features_split{}.npy'
X_train = np.load(train_name.format(train_split))

# read training labels
print("reading labels")
df = pd.read_csv('./train.csv')
y_train = df['Genre'].tolist()
y_train = y_train[:800]
y_train = np.repeat(y_train, train_split)


# normalize data and encode labels
encoder = LabelEncoder()
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
y_train = encoder.fit_transform(y_train)



# gridsearchcv with 5-fold cross validation to find best parameters
param_grid = {
    'C': [.0001, .001, .01, .1, 1, 2, 3, 4, 5, 6, 7, 10, 15],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'degree': [1, 2, 3, 4, 5, 6, 7],
    'gamma': [0.1, 0.01, 0.011, 0.012, 0.015, 0.001]
}

net = SVC(tol=1e-3, class_weight=None, random_state=42)


grid_search = GridSearchCV(net, param_grid, cv=5, scoring='accuracy', verbose=0, n_jobs=-1)

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print("Best Parameters:", best_params)