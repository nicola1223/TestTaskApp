import tensorflow as tf
import tf_keras
import tensorflow_decision_forests as tfdf


def create_model(train_dataset: tf.data.Dataset):
    """
    Function for creation GradientBoostedTree model with tuner
    :return: TensorFlow model
    """
    tuner = tfdf.tuner.RandomSearch(num_trials=50, use_predefined_hps=True)
    model = tfdf.keras.GradientBoostedTreesModel(tuner=tuner, task=tfdf.keras.Task.REGRESSION)
    model.fit(train_dataset, verbose=2)
    return model


def load_model(path):
    return tf_keras.models.load_model(path)
