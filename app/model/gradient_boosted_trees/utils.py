import numpy as np
import pandas as pd
import tensorflow_decision_forests as tfdf


def convert_dataset(data: pd.DataFrame, label: str):
    """
    Function for deleting not important features, normalizing data and converting it to TensorFlow dataset
    :param data: pandas dataset
    :param label: name of column with target
    :return: TensorFlow dataset
    """
    data = data.drop(['Id', 'Open Date', 'City'], axis=1)
    city_groups = data['City Group'].unique().tolist()
    data['City Group'] = data['City Group'].map(city_groups.index)
    types = data['Type'].unique().tolist()
    data['Type'] = data['Type'].map(types.index)
    if label:
        return tfdf.keras.pd_dataframe_to_tf_dataset(data, label=label, task=tfdf.keras.Task.REGRESSION)
    else:
        return tfdf.keras.pd_dataframe_to_tf_dataset(data, task=tfdf.keras.Task.REGRESSION)


def split_dataset(dataset: pd.DataFrame, test_ratio=0.30):
    """
    Function for splitting dataset to train and test datasets
    :param dataset: pandas dataset
    :param test_ratio: ratio for splitting
    :return: tuple of train and test datasets
    """
    test_indices = np.random.rand(len(dataset)) < test_ratio
    return dataset[~test_indices], dataset[test_indices]


def get_datasets_from_csv(path: str):
    """
    Function for getting train and test datasets from csv file
    :param path: path to csv file
    :return: tuple of train and test datasets
    """
    pd_dataset = pd.read_csv(path)
    pd_train_dataset, pd_test_dataset = split_dataset(pd_dataset)
    return convert_dataset(pd_train_dataset, 'revenue'), convert_dataset(pd_test_dataset, 'revenue')