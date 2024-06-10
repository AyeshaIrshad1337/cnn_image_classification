import tensorflow as tf
from utils.data_process import data_load, preprocess, split
from models.model import create_model, train
from utils.plot import plot_accuracy,plot_loss
from utils.evaluation import evaluate
import os
dir = 'data'
image_exts = ['jpeg','jpg', 'bmp', 'png']
data=data_load(dir, image_exts)
data=preprocess(data)
train_data, val_data, test_data=split(data)
model=create_model()
history=train(model, train_data, test_data, val_data)
plot_accuracy(history)
plot_loss(history)
evaluate(model, test_data)
model.save(os.path.join('models','imageclassifier.h5'))