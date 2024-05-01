import librosa
import numpy as np
import os


def extract_features(file, split):
    feats = []
    sig, sr = librosa.load(file, sr=None, mono=True)
    sig = sig[:660000]
    ys = np.array_split(sig, split)
    
    for idx, y in enumerate(ys):
        feats.append([])
        # chroma stft
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma)
        chroma_var = np.var(chroma)
        feats[idx].append(chroma_mean)
        feats[idx].append(chroma_var)


        # rms
        rms = librosa.feature.rms(y=y)
        rms_mean = np.mean(rms)
        rms_var = np.var(rms)
        feats[idx].append(rms_mean)
        feats[idx].append(rms_var)




        # spectral centroid
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_cent_mean = np.mean(spec_cent)
        spec_cent_var = np.var(spec_cent)
        feats[idx].append(spec_cent_mean)
        feats[idx].append(spec_cent_var)


        # spectral bandwidth
        spec_band = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spec_band_mean = np.mean(spec_band)
        spec_band_var = np.var(spec_band)
        feats[idx].append(spec_band_mean)
        feats[idx].append(spec_band_var)


        # rolloff
        spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        spec_rolloff_mean = np.mean(spec_rolloff)
        spec_rolloff_var = np.var(spec_rolloff)
        feats[idx].append(spec_rolloff_mean)
        feats[idx].append(spec_rolloff_var)


        # zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y=y)
        zcr_mean = np.mean(zcr)
        zcr_var = np.var(zcr)
        feats[idx].append(zcr_mean)
        feats[idx].append(zcr_var)

        
        # tempo
        tempo = librosa.feature.tempo(y=y, sr=sr)
        tempo_mean = np.mean(tempo)
        feats[idx].append(tempo_mean)

        

        # mfcc
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_means = np.mean(mfccs, axis=1)
        mfcc_vars = np.var(mfccs, axis=1)
        feats[idx].extend(mfcc_means)
        feats[idx].extend(mfcc_vars)



        # Spectral contrast
        spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        spec_contrast_mean = np.mean(spec_contrast)
        spec_contrast_var = np.mean(spec_contrast)
        feats[idx].append(spec_contrast_mean)
        feats[idx].append(spec_contrast_var) 


        # spectral flatness
        spec_flatness = librosa.feature.spectral_flatness(y=y)
        spec_flatness_mean = np.mean(spec_flatness)
        spec_flatness_var = np.mean(spec_flatness)
        feats[idx].append(spec_flatness_mean)
        feats[idx].append(spec_flatness_var)



        # tonnetz
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        tonnetz_means = np.mean(tonnetz, axis=1)
        tonnetz_vars = np.var(tonnetz, axis=1)
        feats[idx].extend(tonnetz_means)
        feats[idx].extend(tonnetz_vars)



        # return np.array(feat, dtype=np.float64)
    return feats



def add_noise(audio):
    noise = np.random.normal(0, 0.025, len(audio))
    return audio + noise

def write_csv(y_predict, fname):

    y_predict = y_predict.tolist()
    result = [[item] for item in y_predict]

    directory_test = './elec-378-sp2024-final-project/Dataset/test/'
    filenames = os.listdir(directory_test)
    filenames.sort()
    for i, filename in enumerate(filenames):
        result[i] = filename + ',' + y_predict[i]
        


    # Column titles
    column_titles = 'ID,Genre'


    # Combine column titles with data matrix
    result.insert(0, column_titles)
    result = [[item] for item in result]
    result = np.array(result, dtype=str)


    # Save data with titles to CSV file
    np.savetxt((fname + '.csv'), result, delimiter=',', fmt='%s')
    return