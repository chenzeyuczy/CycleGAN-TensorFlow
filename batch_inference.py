"""Translate an image to another image
An example of command-line usage is:
python batch_inference.py --model pretrained/apple2orange.pb \
                       --input_dir input_sample_dir \
                       --output_dir output_sample_dir \
                       --image_size 256
"""

import tensorflow as tf
import os
from model import CycleGAN
import utils
import numpy as np
from PIL import Image
from scipy import misc

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('model', '', 'model path (.pb)')
tf.flags.DEFINE_string('input_dir', 'input_sample_dir', 'directory of input image')
tf.flags.DEFINE_string('output_dir', 'output_sample_dir', 'directory of output image path')
tf.flags.DEFINE_integer('image_size', '256', 'image size, default: 256')
tf.flags.DEFINE_boolean('pad_and_crop', False, 'flag to pad and crop image when IO, default: False')

def inference():
  graph = tf.Graph()
  with graph.as_default():
    with tf.gfile.FastGFile(FLAGS.model, 'rb') as model_file:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(model_file.read())
      tf.import_graph_def(graph_def, name='')

    if not os.path.exists(FLAGS.output_dir):
      os.makedirs(FLAGS.output_dir)
    img_list = os.listdir(FLAGS.input_dir)

    for img_name in img_list:
      img_path = os.path.join(FLAGS.input_dir, img_name)
      img_output = os.path.join(FLAGS.output_dir, img_name)
       
      img = Image.open(img_path)
      img_size = img.size
      input_image = np.array(img)
      img.close()
      if FLAGS.pad_and_crop:
        input_image = utils.image_padding(input_image)
      input_image = misc.imresize(input_image, [FLAGS.image_size, FLAGS.image_size])
     
      config = tf.ConfigProto()
      config.gpu_options.allow_growth = True
      with tf.Session(graph=graph, config=config) as sess:
        output_image = sess.run(['output_image:0'], feed_dict={'input_image:0': input_image})
        generated = output_image[0]
      with open(img_output, 'wb') as f:
        f.write(generated)
  
        # Resize image to its original size.
        img = Image.open(img_output)
      if FLAGS.pad_and_crop:
        img_input = np.array(img)
        crop_width = min(img_size)
        img_input = utils.image_cropping(img_input, crop_width)
        img = Image.fromarray(img_input)
      img.resize(img_size).save(img_output)
      img.close()

def main(unused_argv):
  inference()

if __name__ == '__main__':
  tf.app.run()
