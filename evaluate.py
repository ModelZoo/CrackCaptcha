from model_zoo.evaluater import BaseEvaluater
import tensorflow as tf
from model_zoo.trainer import BaseTrainer
import numpy as np
from os import listdir
from os.path import join, exists
from sklearn.model_selection import train_test_split
import cv2

tf.flags.DEFINE_string('dataset', 'dun163', help='Dataset')
tf.flags.DEFINE_string('checkpoint_name', 'model.ckpt-20', help='Model name')
tf.flags.DEFINE_string('datasets_dir', './datasets', help='Data dir')
tf.flags.DEFINE_integer('image_width', 600, help='Image width')
tf.flags.DEFINE_integer('image_height', 300, help='Image height')


class Evaluater(BaseEvaluater):
    
    def generate_data(self):
        """
        build generator of data
        :return:
        """
        # read data
        datasets_dir = self.flags.datasets_dir
        dataset = self.flags.dataset
        dataset_dir = join(datasets_dir, dataset)
        count = 0
        # get all labeled data
        for file in listdir(dataset_dir):
            count += 1
            if file.endswith('.txt'):
                if count % 100 == 0:
                    print('Count', count)
                if count > 40:
                    break
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
        # x_train, x_eval, y_train, y_eval = train_test_split(x_data, y_data, test_size=0.33, random_state=42)
        print(y_data.shape)
        print('Yata', y_data)
        # print('Sample', x_train[0], y_train[0], x_train.dtype, y_train.dtype)
        # print('X Data Shape', x_data.shape, 'Y Data Shape', y_data.shape)
        return x_data, y_data


if __name__ == '__main__':
    result = Evaluater().run()
    print(result)
