from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
def evaluate(model, test):
    """
    Evaluate the model.
    Args:
        model (tf.keras.Model): The model to evaluate.
        test (tf.data.Dataset): The test data.
    Returns:
        No
    """
    model.evaluate(test)
    precision = Precision()
    recall = Recall()
    accuracy = BinaryAccuracy()
    for x, y in test:
        model.predict(x)
        precision.update_state(y, model.predict(x))
        recall.update_state(y, model.predict(x))
        accuracy.update_state(y, model.predict(x))
    print('Precision: {}'.format(precision.result().numpy()))
    print('Recall: {}'.format(recall.result().numpy()))
    print('Accuracy: {}'.format(accuracy.result().numpy()))
    