from tensorflow.python import pywrap_tensorflow
import numpy as np
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string('input_path', '/data/object_detection-pretrained-ckpt/faster_rcnn_resnet101_coco_11_06_2017/model.ckpt',
                    'path of pretrained_checkpoint')
flags.DEFINE_string('output_path', 'ckpts/model.ckpt', 'output checkpoint')
flags.DEFINE_string('feature_extractor', 'resnet_v1_101', 'name of first checkpoint')
flags.DEFINE_string('num_input_channels', 6, 'number of input channel. Each image, background, diff image require 3 channels')
flags.DEFINE_string('edit_method', 'spread', 'divide the checkpoint convolution variable by the number of channels'
                                             ' divided by 3 and clone it to every set of 3 channels. random: initialize'
                                             ' extra channels feature map to random truncated_normal with sttdev=0.2'
                                             '. clone: clone the value to new channels')
FLAGS = flags.FLAGS

if __name__ == '__main__':
  reader = pywrap_tensorflow.NewCheckpointReader(FLAGS.input_path)
  var_to_shape_map = reader.get_variable_to_shape_map()
  var_to_edit_names = ['FirstStageFeatureExtractor/{}/conv1/weights'.format(FLAGS.feature_extractor),
                       'FirstStageFeatureExtractor/{}/conv1/weights/Momentum'.format(FLAGS.feature_extractor),]
  print('Loading checkpoint...')
  for key in sorted(var_to_shape_map):
    if key not in var_to_edit_names:
      var = tf.Variable(reader.get_tensor(key), name=key, dtype=tf.float32)
    else:
      print("Found variable: {}".format(key))
  vars_to_edit = []
  for name in var_to_edit_names:
    if reader.has_tensor(name):
      vars_to_edit.append(reader.get_tensor(name))
    else:
      raise Exception("{} not found in checkpoint. Check feature extractor name. Exiting.".format(name))
  new_vars = []
  sess = tf.Session()
  for name, var_to_edit in zip(var_to_edit_names, vars_to_edit):
    if FLAGS.edit_method in ['spread', 'clone']:
      checkpoint_num_input_channels = var_to_edit.shape[2]
      if FLAGS.num_input_channels % checkpoint_num_input_channels != 0:
        raise Exception('For spread edit method, num_input_channels must be divisible by num input channels of checkpoint!')
      num_clones = int(FLAGS.num_input_channels / checkpoint_num_input_channels)
      if FLAGS.edit_method == 'spread':
        spreaded_var = var_to_edit / num_clones
      else:
        spreaded_var = var_to_edit
      new_var = np.tile(spreaded_var, [1, 1, num_clones, 1])
      new_vars.append(tf.Variable(new_var, name=name, dtype=tf.float32))
    elif FLAGS.edit_method == 'random':
      random_shape = list(var_to_edit.shape)
      random_shape[2] = FLAGS.num_input_channels - 3
      random_var = tf.truncated_normal(shape=random_shape, stddev=0.01).eval(session=sess)
      new_var = np.concatenate([var_to_edit, random_var], axis=2)
      new_vars.append(tf.Variable(new_var, name=name, dtype=tf.float32))
    else:
      raise Exception("Edit method must be spread or zeros or clone!")
  sess.run(tf.global_variables_initializer())
  saver = tf.train.Saver()
  saver.save(sess, FLAGS.output_path)

#Only need .0000-of-0001 and .index file. Good to go!
