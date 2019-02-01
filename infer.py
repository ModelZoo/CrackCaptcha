from model_zoo.inferer import BaseInferer
import tensorflow as tf
import cv2
from os import listdir
from os.path import join
import numpy as np

tf.flags.DEFINE_string('checkpoint_name', 'model.ckpt-12', help='Model name')
tf.flags.DEFINE_string('test_dir', 'tests/dun163', help='Test dir')
tf.flags.DEFINE_integer('image_width', 600, help='Image width')
tf.flags.DEFINE_integer('image_height', 300, help='Image height')


class Inferer(BaseInferer):
    
    def prepare_data(self):
        """
        prepare test data
        :return:
        """
        test_dir = self.flags.test_dir
        items = sorted(list(listdir(test_dir)))
        items_path = list(map(lambda x: join(test_dir, x), items))
        test_data = list(map(lambda x: self.process_image(x), items_path))
        test_data = np.asarray(test_data, dtype=np.float32)
        test_data /= 255.0
        print('Test data shape', test_data.shape)
        self.items = items
        return test_data
    
    def process_image(self, image_file):
        """
        read image by cv2
        :param image_file:
        :return:
        """
        image = cv2.imread(image_file)
        image = cv2.resize(image, (self.flags.image_height, self.flags.image_width))
        return image.tolist()


if __name__ == '__main__':
    inferer = Inferer()
    logits = inferer.run()
    for item, logit in zip(inferer.items, logits):
        print('=' * 20)
        print('Image Path:', item)
        print('Predict Result:', logit)
