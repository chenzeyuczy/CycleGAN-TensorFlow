"""Translate an image to another image
An example of command-line usage is:
python export_graph.py --model pretrained/apple2orange.pb \
                       --input input_sample.jpg \
                       --output output_sample.jpg \
                       --image_size 256
"""

import tensorflow as tf
import os
from model import CycleGAN
import utils

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('model', '', 'model path (.pb)')
tf.flags.DEFINE_string('input_dir', 'input_sample_dir', 'directory of input image')
tf.flags.DEFINE_string('output_dir', 'output_sample_dir', 'directory of output image path')
tf.flags.DEFINE_integer('image_size', '256', 'image size, default: 256')

def inference():
  graph = tf.Graph()
  with graph.as_default():
    with tf.gfile.FastGFile(FLAGS.model, 'rb') as model_file:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(model_file.read())

    img_list = os.listdir(FLAGS.input_dir)
    for img_name in img_list:
      img_path = os.path.join(FLAGS.input_dir, img_name)
      img_output = os.path.join(FLAGS.output_dir, img_name)
      if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)
      
        
      with tf.gfile.FastGFile(img_path, 'rb') as f:
        image_data = f.read()
        input_image = tf.image.decode_jpeg(image_data, channels=3)
        input_image = tf.image.resize_images(input_image, size=(FLAGS.image_size, FLAGS.image_size))
        input_image = utils.convert2float(input_image)
        input_image.set_shape([FLAGS.image_size, FLAGS.image_size, 3])
    
      [output_image] = tf.import_graph_def(graph_def,
                              input_map={'input_image': input_image},
                              return_elements=['output_image:0'],
                              name='output')
    
      with tf.Session(graph=graph) as sess:
        generated = output_image.eval()
        with open(img_output, 'wb') as f:
          f.write(generated)

def main(unused_argv):
  inference()

if __name__ == '__main__':
  tf.app.run()
