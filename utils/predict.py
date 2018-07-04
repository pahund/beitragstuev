import pandas as pd
import tensorflow as tf


def predict(sentence, estimator):
    data = pd.DataFrame.from_dict({"sentence": [sentence]})
    input_fn = tf.estimator.inputs.pandas_input_fn(data, shuffle=False)
    return next(estimator.predict(input_fn=input_fn, predict_keys=["classes", "probabilities"]))

def classify(prediction):
    return "positive" if prediction["classes"][0] == b'1' else "negative"

def rate(prediction):
    return int(round(prediction["probabilities"][1] * 100))

def print_analysis(sentence, estimator):
    print(sentence)
    prediction = predict(sentence, estimator)
    print(classify(prediction))
    print(rate(prediction))
