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
print("processing testing data")


data_test = []

# path
directory = './elec-378-sp2024-final-project/Dataset/test/'
i = 0
filenames = os.listdir(directory)
filenames.sort()
# iterate through all files
for filename in filenames:
    f = os.path.join(directory, filename)
    res = utils.extract_features(f, split)
    for s in range(split):
        data_test.append(res[s])
    
    i += 1

    print(i)
    # if (i % 50 == 0):
    #     print(i)

# Convert list to numpy array
#data_list = np.vstack(data_list)
data_test = np.array(data_test, dtype=np.float64)
print(f"Type: {type(data_test)}, Data shape: {data_test.shape}")

# np.save('features_test_split10', data_test)
savename = 'features_test_split{}_all_nmfcc20'
np.save(savename.format(split), data_test)
