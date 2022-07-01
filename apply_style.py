import os
import cv2.cv2
import tensorflow as tf
# Load compressed models from tensorflow_hub
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'
import IPython.display as display
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False
import numpy as np
import PIL.Image
import time
import functools
import tensorflow_hub as hub
from os import listdir
from os.path import isfile, join




###----------------------------------------- FUNCTIONS -----------------------------------------------------------------

def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return PIL.Image.fromarray(tensor)


def load_img(path_to_img):
  max_dim = 512
  img = tf.io.read_file(path_to_img)
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)

  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  long_dim = max(shape)
  scale = max_dim / long_dim

  new_shape = tf.cast(shape * scale, tf.int32)

  img = tf.image.resize(img, new_shape)
  img = img[tf.newaxis, :]
  return img


def imshow(image, title=None):
  if len(image.shape) > 3:
    image = tf.squeeze(image, axis=0)
  plt.imshow(image)
  if title:
    plt.title(title)
  plt.show()


def stylize_image(content_im, style_im, hub_model):
  # plt.subplot(1, 2, 1)
  # imshow(content_im, 'Content Image')
  # plt.subplot(1, 2, 2)
  # imshow(style_im, 'Style Image')
  stylized_tensor = hub_model(tf.constant(content_im), tf.constant(style_im))[0]
  stylized_im = tensor_to_image(stylized_tensor)
  #imshow(stylized_image, 'Stylized Image')
  return stylized_im


def stylize_images_set(content_set, style_set):
  content_path = './content_images'
  style_path = './style_images'
  stylized_path = './stylized_images'

  # sets directories
  content_set_path= os.path.join(content_path, content_set)
  style_set_path= os.path.join(style_path, style_set)
  stylized_set_path= os.path.join(stylized_path, content_set +'_'+ style_set)
  if not os.path.exists(stylized_set_path):
    os.mkdir(stylized_set_path)

  # get files
  content_files = [f for f in listdir(content_set_path) if isfile(join(content_set_path, f))]
  style_files = [f for f in listdir(style_set_path) if isfile(join(style_set_path, f))]

  # load model
  hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')


  for content_file in content_files:
    content_im_path = os.path.join(content_set_path, content_file)
    content_im = load_img(content_im_path)

    for style_file in style_files:
      style_im_path = os.path.join(style_set_path, style_file)
      style_im = load_img(style_im_path)

      # call the model
      stylized_im = stylize_image(content_im, style_im, hub_model)

      # save_im
      stylized_file = content_file[:-4] + '_' + style_file[:-4]+'.jpg'
      stylized_im_path = os.path.join(stylized_set_path, stylized_file)
      stylized_im.save(stylized_im_path)

      #cv2.imshow('im', stylized_im)
      #cv2.waitKey(200)



#-------------------------------------------------- MAIN ---------------------------------------------------------------
style_set='set1'

#content_set='set1'
#stylize_images_set(content_set, style_set)

content_set='perspective_transformation'
stylize_images_set(content_set, style_set)

content_set='black_background'
stylize_images_set(content_set, style_set)

#-------------------------------------------------------------------------------------------------------

