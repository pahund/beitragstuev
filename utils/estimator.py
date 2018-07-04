import tensorflow as tf
import tensorflow_hub as hub


def create(text_embedding_module):
    embedded_text_feature_column = hub.text_embedding_column(
        key="sentence",
        module_spec=text_embedding_module)

    return tf.estimator.DNNClassifier(
        hidden_units=[500, 100],
        feature_columns=[embedded_text_feature_column],
        n_classes=2,
        optimizer=tf.train.AdagradOptimizer(learning_rate=0.003))


def train(estimator, data, steps):
    train_input_fn = tf.estimator.inputs.pandas_input_fn(data, data["polarity"], num_epochs=None, shuffle=True)
    estimator.train(input_fn=train_input_fn, steps=steps)


def create_and_train(data, text_embedding_module, steps):
    estimator = create(text_embedding_module)
    train(estimator, data, steps)
    return estimator


def save(estimator, text_embedding_module, export_dir_base):
    embedded_text_feature_column = hub.text_embedding_column(
        key="sentence",
        module_spec=text_embedding_module)
    feature_columns = [embedded_text_feature_column]
    feature_spec = tf.feature_column.make_parse_example_spec(feature_columns)
    serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
    estimator.export_savedmodel(export_dir_base=export_dir_base,
                                serving_input_receiver_fn=serving_input_receiver_fn)
