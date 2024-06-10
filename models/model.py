import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
def create_model():
    """
    Create a model
    Args: No
    Returns:
        model (tf.keras.Model): The model.
    """
    model=Sequential([
        Conv2D(16,3, padding='same', activation='relu', input_shape=(224,224,3)),
        MaxPooling2D(),
        Conv2D(32, 3, padding='same',activation='relu'), 
        MaxPooling2D(),
        Conv2D(64,3, padding='same', activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(1, activation='sigmoid')
        ])
    model.compile(optimizer='adam',loss=tf.keras.losses.BinaryCrossentropy(),metrics=['accuracy'])
    print(model.summary())
    return model
def train(model, train, test, val):
    """
    Train the model.
    Args:
        model (tf.keras.Model): The model to train.
        train (tf.data.Dataset): The training data.
        test (tf.data.Dataset): The test data.
        val (tf.data.Dataset): The validation data.
    Returns:
        history (tf.keras.History): The history of the training.
    """
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='logs')
    history=model.fit(train, validation_data=val, epochs=20, callbacks=[tensorboard_callback])
    return history