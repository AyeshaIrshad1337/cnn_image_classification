import cv2
import os
import imghdr
import tensorflow as tf
def data_load(dir, ext):
    """
    Load data from a directory with a specific extension.
    Args:
        dir (str): The directory to load the data from.
        ext (str): The extension of the files to load.
    Returns:
        data (list): A list of dataframes.
    """
    for image_class in os.listdir(dir):
        for image in os.listdir(os.path.join(dir, image_class)):
            img_path=os.path.join(dir, image_class, image)
            try:
                img=cv2.imread(img_path)
                tip=imghdr.what(img_path)
                if tip not in ext:
                    print('Image not in ext list {}'.format(img_path))
                    os.remove(img_path) 
            except:
                print('Image not found {}'.format(img_path))
    data = tf.keras.preprocessing.image_dataset_from_directory(dir)  
    return data                  
def preprocess(data):
    data=data.map(lambda x, y: (tf.image.resize(x, (224, 224)), y))
    data=data.map(lambda x, y: (x/255, y))
    return data
def split(data):
    train_size = int(0.8 * len(data))
    val_size = int(0.1 * len(data))
    test_size = int(0.1 * len(data))
    train = data.take(train_size)
    test = data.skip(train_size)
    val = test.skip(test_size)
    test = test.take(test_size)
    return train, val, test