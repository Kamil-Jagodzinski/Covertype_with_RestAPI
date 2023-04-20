import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def load_data():
    path = "./covtype.data.gz"
    # Preparing columns labels before reading dataset
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
    column_names.append("Cover_Type")

    # Push dataset into pandas table
    print("============ Downloading dataset ============")
    dataset = pd.read_csv(path, header=None, names=column_names)
    dataset["Cover_Type"] = dataset["Cover_Type"] - 1
    return dataset

def preprocess_data(dataset):
    # Split into data and labels
    x = dataset.drop(dataset.columns[[54]],axis=1)
    y = dataset.iloc[:,54]

    # Split into train and test datasets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=25,stratify=y)
    return x_train, x_test, y_train, y_test

def preprocess_one_hot_data(dataset):
    # Split into data and labels
    x = dataset.drop(dataset.columns[[54]],axis=1)
    y = dataset.iloc[:,54]
    y_one_hot = to_categorical(y)

    # Split into train and test datasets
    return x, y_one_hot
