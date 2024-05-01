from sklearn.feature_selection import mutual_info_classif
import numpy as np
import pandas as pd
import pprint


# amount of times to split data
train_split = 5
test_split = 5


# read training data
print("reading features")
train_name = 'features_split{}_all_nmfcc20.npy'
X_train = np.load(train_name.format(train_split))


# read training labels
print("reading labels")
df = pd.read_csv('./train.csv')
y_train = df['Genre'].tolist()
y_train = y_train[:800]
y_train = np.repeat(y_train, train_split)


# Create vector of feature names
feats = ['chroma_stft_mean', 'chroma_stft_std', 'rms_mean', 'rms_std', 'spectral_centroid_mean', 
         'spectral_centroid_std', 'spectral_bandwidth_mean', 'spectral_bandwidth_std',
         'spectral_rolloff_mean', 'spectral_rolloff_std', 'zero_crossing_rate_mean',
         'zero_crossing_rate_std', 'tempo_mean']
for i in range(20):
    feats.append('mfcc_{}_mean'.format(i))
for i in range(20):
    feats.append('mfcc_{}_std'.format(i))
feats.append('spectral_contrast_mean')
feats.append('spectral_contrast_std')
feats.append('spectral_flatness_mean')
feats.append('spectral_flatness_std')
for i in range(6):
    feats.append('tonnetz_{}_mean'.format(i))  
for i in range(6):
    feats.append('tonnetz_{}_std'.format(i))  


# compute Mutual information between labels and features
res = mutual_info_classif(X_train, y_train)

igr_labels = dict(zip(feats, res))
igr_cols = dict(zip(range(len(feats)), res))

# Print features, sorted by mutual information
d_view = sorted( ((v,k) for k,v in igr_labels.items()), reverse=True) 
for v,k in d_view:
    print (k, ":\t\t\t\t\t", v)


print("\n\n\n\n-----------------------------------")


# print feature columns, sorted by mutual information
# this is useful since we can just delete the lowest columns from the
# data matrix
d_view2 = sorted( ((v,k) for k,v in igr_cols.items()), reverse=True) 
for v,k in d_view2:
    print (k, ":\t\t\t\t\t", v)