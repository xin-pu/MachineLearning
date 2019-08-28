import numpy as np
from sklearn import preprocessing

data = np.array([[3, -1.5, 2, -5.4],
                 [0, 4, -0.3, 2.1],
                 [1, 3.3, -1.9, -4.3]])
print(data)


# Step 1 Mean removal
def mean():
    data_standardized = preprocessing.scale(data)
    data_mean = data_standardized.mean(axis=0)
    data_std = data_standardized.std(axis=0)
    print(data_standardized)
    print(data_mean)
    print(data_std)

# Step 2 Scalling
def scaling():
    data_scaler=preprocessing.MinMaxScaler(feature_range=(0,1))
    data_scaled=data_scaler.fit_transform(data)
    print(data_scaled)

# Step 3 Normalization
def normalization():
    data_normalized=preprocessing.normalize(data,norm='l1')
    print(data_normalized)

# Step 4 Binarization
def binarization():
    data_binarized=preprocessing.Binarizer(threshold=1.4)
    print(data_binarized.transform(data))
