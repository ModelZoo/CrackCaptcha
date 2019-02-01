from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

image_generator = ImageDataGenerator(
    height_shift_range=0.1,
    channel_shift_range=100,
    vertical_flip=True,
)

image = load_img('./datasets/dun163/captcha.png')
x = img_to_array(image)
x = x.reshape((1,) + x.shape)

i = 0
for batch in image_generator.flow(x,
                                  batch_size=1,
                                  save_to_dir='./enhances',
                                  save_prefix='',
                                  save_format='jpg'):
    print(batch)
