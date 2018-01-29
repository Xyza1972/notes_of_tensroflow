import argparse
import sys
import tempfile

import numpy as np
import tensorflow as tf
import inception_resnet_v2

slim = tf.contrib.slim

FLAGS = None

# Enable logging
tf.logging.set_verbosity(tf.logging.INFO)

# function to load data
#def get_dataset():


# Build model function for Estimator
def model_fn(features, labels, mode, params):
    # Logic to do the following:
    # 1. Configure the model via TensorFlow operations
    # 2. Define the loss function for training/evaluation
    # 3. Define the training operation/optimizer
    # 4. Generate predictions
    # 5. Return predictions/loss/train_op/eval_metric_ops in EstimatorSpec object

    first_hidden_layer = tf.layers.dense(features['x'], 10, activation=tf.nn.relu)
    second_hidden_layer = tf.layers.dense(first_hidden_layer, 10, activation=tf.nn.relu)
    output_layer = tf.layers.dense(second_hidden_layer, 1)

    predictions = tf.reshape(output_layer, [-1])

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
        predictions={"ages": predictions})

    loss = tf.losses.mean_squared_error(labels, predictions)

    eval_metric_ops = {
        "rmse": tf.metrics.root_mean_squared_error(
            tf.cast(labels, tf.float64), predictions)
    }

    optimizer = tf.train.GradientDescentOptimizer(
        learning_rate=params["learning_rate"])
    train_op = optimizer.minimize(
        loss=loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode=mode,
                                      predictions=predictions,
                                      loss=loss,
                                      train_op=train_op,
                                      eval_metric_ops=eval_metric_ops)


# Load data into Datasets
# Define flags to use specify files for training, test and prediction
def main(unused_argv):
    # Load datasets here
    #  code for generate datasets
    #
    print ("start")

    abalone_train = './abalone_train.csv'
    abalone_test = './abalone_test.csv'
    abalone_predict = './abalone_predict.csv'

    training_set = tf.contrib.learn.datasets.base.load_csv_without_header(
        filename=abalone_train, target_dtype=np.int, features_dtype=np.float64)
    test_set = tf.contrib.learn.datasets.base.load_csv_without_header(
        filename=abalone_test, target_dtype=np.int, features_dtype=np.float64)
    prediction_set = tf.contrib.learn.datasets.base.load_csv_without_header(
        filename=abalone_predict, target_dtype=np.int, features_dtype=np.float64)

    model_params = {'learning_rate': 0.001}

    nn = tf.estimator.Estimator(model_fn=model_fn, model_dir='./model', params=model_params)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(training_set.data)},
        y=np.array(training_set.target),
        num_epochs=None,
        shuffle=True)

    nn.train(input_fn=train_input_fn, steps=5000)

    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(test_set.data)},
        y=np.array(test_set.target),
        num_epochs=1,
        shuffle=False)
    ev = nn.evaluate(input_fn=test_input_fn)

    print("Loss: %s" % ev["loss"])
    print("Root Mean Squared Error: %s" % ev["rmse"])



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v:v.lower() == 'true')
    parser.add_argument(
        "--train_data", type=str, default="", help="Path to the training data.")
    parser.add_argument(
        "--test_data", type=str, default="", help="Path to the test data.")
    parser.add_argument(
        "--predict_data", type=str, default="", help="Path to the prediction data.")
    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
