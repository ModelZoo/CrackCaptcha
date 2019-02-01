import tensorflow as tf
from model_zoo.trainer import BaseTrainer
import numpy as np
from os import listdir
from os.path import join, exists
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image
import cv2

tf.flags.DEFINE_string('dataset', 'dun163', help='Dataset')
tf.flags.DEFINE_string('datasets_dir', './datasets', help='Data dir')
tf.flags.DEFINE_float('learning_rate', 0.0001, help='Learning rate')
tf.flags.DEFINE_integer('image_width', 300, help='Image width')
tf.flags.DEFINE_integer('image_height', 150, help='Image height')
tf.flags.DEFINE_integer('epochs', 1000, help='Max epochs')
tf.flags.DEFINE_integer('early_stop_patience', 500, help='Early stop patience')
tf.flags.DEFINE_bool('checkpoint_restore', True, help='Model restore')
tf.flags.DEFINE_string('model_class', 'VGGModel', help='Model restore')
tf.flags.DEFINE_integer('batch_size', 10, help='Batch size')
tf.flags.DEFINE_integer('checkpoint_save_freq', 1, help='Save model every epoch number')
tf.flags.DEFINE_integer('enhance_images_number', 20, help='Enhance images number')


class Trainer(BaseTrainer):
    image_generator = image.ImageDataGenerator(
        height_shift_range=0.1,
        channel_shift_range=100,
        vertical_flip=True,
    )
    
    def enhance_images(self, x_data, y_data):
        """
        generate enhanced image
        :param image:
        :return:
        """
        x_data_enhanced, y_data_enhanced = [], []
        for x, y in zip(x_data, y_data):
            # add original data
            x_data_enhanced.append(x)
            y_data_enhanced.append(y)
            # add enhanced data
            image = np.expand_dims(x, axis=0)
            gen = self.image_generator.flow(image)
            for i in range(self.flags.enhance_images_number):
                enhanced_image = next(gen)
                enhanced_image = np.reshape(enhanced_image, enhanced_image.shape[1:])
                x_data_enhanced.append(enhanced_image)
                y_data_enhanced.append(y)
        return np.asarray(x_data_enhanced), np.asarray(y_data_enhanced)
    
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
        for file in listdir(dataset_dir):
            if file.endswith('.txt'):
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
        
        print('Generated data', len(x_data), len(y_data))
        
        x_data, y_data = np.asarray(x_data, dtype=np.float32), np.asarray(y_data, dtype=np.float32)
        x_train, x_eval, y_train, y_eval = train_test_split(x_data, y_data, test_size=0.20, random_state=42)
        
        print('Enhancing images', x_train.shape)
        
        x_train, y_train = self.enhance_images(x_train, y_train)
        x_train /= 255.0
        
        print('Sample', x_train[0], y_train[0], x_train.dtype, y_train.dtype)
        print('X Train Data Shape', x_train.shape, 'Y Data Shape', y_train.shape)
        return (x_train, y_train), (x_eval, y_eval)


if __name__ == '__main__':
    Trainer().run()
