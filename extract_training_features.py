import numpy as np
from scipy.io import wavfile
from scipy.fft import fft, ifft
import librosa
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import utils

TEST = False
split = 5

###################################################################################
# PROCESS DATA
###################################################################################
print("processing training data")

data = []

# path
directory = './elec-378-sp2024-final-project/Dataset/train/'
i = 0
filenames = os.listdir(directory)
filenames.sort()
# iterate through all files
for filename in filenames:
    f = os.path.join(directory, filename)
    res = utils.extract_features(f, split)
    for s in range(split):
        data.append(res[s])
    
    i += 1

    print(i)
    # if (i % 50 == 0):
    #     print(i)

# Convert list to numpy array
#data_list = np.vstack(data_list)
data = np.array(data, dtype=np.float64)
print(f"Type: {type(data)}, Data shape: {data.shape}")

savename = 'features_split{}_all_nmfcc20'
np.save(savename.format(split), data)

