import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
import os.path

def load_data():
    path = "./data/covtype.data.gz"
    # Preparing columns labels before reading dataset
    cn = prepare_columns_names()
    cn.append("Cover_Type")

    # Push dataset into pandas table
    dataset = pd.read_csv(path, header=None, names=cn)
    dataset["Cover_Type"] = dataset["Cover_Type"] - 1

    return dataset

def preprocess_data(dataset):
    # Split into data and labels
    x = dataset.drop(dataset.columns[[54]],axis=1)
    y = dataset.iloc[:,54]

    # Split into train and test datasets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y)
    return x_train, x_test, y_train, y_test

def preprocess_one_hot_data(dataset):
    # Split into data and labels
    x = dataset.drop(dataset.columns[[54]],axis=1)
    y = dataset.iloc[:,54]
    y_one_hot = to_categorical(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y_one_hot, test_size=0.2, stratify=y_one_hot)

    if os.path.exists('models/pca.joblib'):
        pca = load_pca('models/pca.joblib')
    else:
        pca = PCA(n_components=16)
        pca.fit(x_train)
        dump(pca, 'models/pca.joblib')

    if os.path.exists('models/std_scaler.joblib'):
        std_scaler = load_scalar('models/std_scaler.joblib')
    else:
        std_scaler = StandardScaler()
        std_scaler.fit(x_train)
        dump(std_scaler, 'models/std_scaler.joblib')

    x_train = pca.fit_transform(x_train)
    x_test = pca.transform(x_test)

    return x_train, x_test, y_train, y_test

def prepare_columns_names():
    column_names = [
        "Elevation",
        "Aspect",
        "Slope",
        "Horizontal_Distance_To_Hydrology",
        "Vertical_Distance_To_Hydrology",
        "Horizontal_Distance_To_Roadways",
        "Hillshade_9am",
        "Hillshade_Noon",
        "Hillshade_3pm",
        "Horizontal_Distance_To_Fire_Points",
    ]
    # Add columns labels with same names but different number   
    for i in range(1,5):
        column_names.append(f"Wilderness_Area{i}")
    for i in range(1,41):
        column_names.append(f"Soil_type{i}")
    return column_names

def load_scalar(path = "models/std_scaler.joblib"):
   scalar = load(path)
   return scalar

def load_pca(path = "models/pca.joblib"):
   pca = load(path)
   return pca
