import tensorflow as tf
from model_zoo.trainer import BaseTrainer
import numpy as np
from os import listdir
from os.path import join, exists
from sklearn.model_selection import train_test_split
import cv2

tf.flags.DEFINE_string('dataset', 'dun163', help='Dataset')
tf.flags.DEFINE_string('datasets_dir', './datasets', help='Data dir')
tf.flags.DEFINE_float('learning_rate', 0.001, help='Learning rate')
tf.flags.DEFINE_integer('image_width', 600, help='Image width')
tf.flags.DEFINE_integer('image_height', 300, help='Image height')
tf.flags.DEFINE_integer('epochs', 1000, help='Max epochs')
tf.flags.DEFINE_integer('early_stop_patience', 500, help='Early stop patience')
tf.flags.DEFINE_bool('checkpoint_restore', True, help='Model restore')
tf.flags.DEFINE_string('model_class', 'VGGModel', help='Model restore')
tf.flags.DEFINE_integer('batch_size', 5, help='Batch size')
tf.flags.DEFINE_integer('checkpoint_save_freq', 2, help='Save model every epoch number')


class Trainer(BaseTrainer):
    def generate_data(self):
        """
        build generator of data
        :return:
        """
        # read data
        datasets_dir = self.flags.datasets_dir
        dataset = self.flags.dataset
        dataset_dir = join(datasets_dir, dataset)
        # get all labeled data
        count = 0
        for file in listdir(dataset_dir):
            if file.endswith('.txt'):
                count += 1
                if count % 500 == 0:
                    print('Processed', count, 'Files')
                image_path = join(datasets_dir, dataset, file.replace('.txt', '.png'))
                label_path = join(datasets_dir, dataset, file)
                if exists(image_path) and exists(label_path):
                    image = cv2.imread(image_path)
                    image = cv2.resize(image, (self.flags.image_height, self.flags.image_width))
                    label = float(open(label_path).read().strip())
                    yield image.tolist(), label
    
    def prepare_data(self):
        """
        prepare data for training
        :return:
        """
        x_data, y_data = [], []
        print('Generating data...')
        for image, label in self.generate_data():
            x_data.append(image)
            y_data.append(label)
        
        x_data, y_data = np.asarray(x_data, dtype=np.float32), np.asarray(y_data, dtype=np.float32)
        
        x_data /= 255.0
        print('X Data Shape', x_data.shape, 'Y Data Shape', y_data.shape)
        
        x_train, x_eval, y_train, y_eval = train_test_split(x_data, y_data, test_size=0.33, random_state=42)
        print('Sample', x_train[0], y_train[0], x_train.dtype, y_train.dtype)
        return (x_train, y_train), (x_eval, y_eval)


if __name__ == '__main__':
    Trainer().run()
