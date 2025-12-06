#!/usr/bin/python

import tensorflow as tf
import numpy as np
import os
import scipy.io
import csv
import pickle
import sys
import subprocess

from network_structure import Model_structure
from data_manager import DataManager

# choose run mode ('debug' or 'user')
run_mode = sys.argv[1]
fname = sys.argv[2]

flags = tf.app.flags
FLAGS = flags.FLAGS

model_dir = 'expr/lr0.0003_dr0.9_nx3_ny3_xy3_yx1_x0.2_y0.2_K10_ba400'

flags.DEFINE_integer('num_layer_x', 3, 'X embedding network layers')
flags.DEFINE_integer('num_layer_y', 3, 'Y embedding network layers')
flags.DEFINE_integer('constraint_xy', 3, 'Constraint Weight xy')
flags.DEFINE_integer('constraint_yx',1, 'Constraint Weight yx')
flags.DEFINE_integer('constraint_x', 0.2, 'Constraint Structure Weight x')
flags.DEFINE_integer('constraint_y', 0.2, 'Constraint Structure Weight y')
flags.DEFINE_integer('top_K', 3, 'Top mosk K number for violation')
flags.DEFINE_integer('test_batch_size', 10, 'Test batch size.')
flags.DEFINE_string('summaries_dir', model_dir, 'Directory to put the summary and log data.')

net_opts = Model_structure.OPTS()
net_opts.network_name = 'Wrapping Network'
net_opts.x_dim = 1140
net_opts.y_dim = 1024
net_opts.x_num_layer = FLAGS.num_layer_x
net_opts.y_num_layer = FLAGS.num_layer_y
net_opts.constraint_weights = [FLAGS.constraint_xy, FLAGS.constraint_yx, FLAGS.constraint_x, FLAGS.constraint_y]
net_opts.is_linear = True
net = Model_structure(net_opts)
net.construct()

saver = tf.train.Saver(tf.global_variables())
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  checkpoint_dir = os.path.join(FLAGS.summaries_dir, 'checkpoints')
  checkpoint_prefix = os.path.join(checkpoint_dir, "model.ckpt")
  if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

  step = 0
  ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
  if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
    #saver.restore(sess, os.path.join(checkpoint_dir, 'model.ckpt-9399'))
    step = int(ckpt.model_checkpoint_path.split('-')[-1])
    step += 1
    print('Session restored successfully. step: {0}'.format(step))

  num_test = FLAGS.test_batch_size
  data_manager = DataManager('/scratch/groups/rwr/joswong/CS230/VM-NET/data/') 
  test_batch = data_manager.batch_iterator_thread(FLAGS.test_batch_size, is_train=False)
  x_batch, y_batch, aff_xy, name_batch = test_batch.next()

  if run_mode == 'user': 
     x_batch, name_batch = data_manager.deploy()
     
     try:
        y_batch = np.array([pickle.load(open(fname, 'rb'))])
     except:
        raise ValueError('invalid file path')

     xy, yx, xy_idx, yx_idx = sess.run([net.recall_xy, net.recall_yx, net.xy_idx, net.yx_idx], feed_dict={
         net.x_data: x_batch,
         net.y_data: y_batch,
         net.K: int(1),
         net.aff_xy: np.zeros((len(x_batch), 1)),
         net.keep_prob: 1., net.is_training: False})

     print('Best Music file: %s' % str(name_batch[yx_idx[0]][0]))
     subprocess.check_output('cp data/audio/%s %s' % (str(name_batch[yx_idx[0]][0]).replace('.pkl', '.mp3'), fname.replace('_feature.pkl', '.mp3')), shell=True)
 
  if run_mode == 'debug1': 
     x_batch, name_batch = data_manager.deploy()

     for i in range(1,400):
        fname = '%03d_frnt.pkl' % i
        print(fname)

        try:
           if os.path.exists('data/video_feature/train/%s' % fname):
              y_batch = np.array([pickle.load(open('data/video_feature/train/%s' % fname, 'rb'))])
           elif os.path.exists('data/video_feature/test/%s' % fname):
              y_batch = np.array([pickle.load(open('data/video_feature/test/%s' % fname, 'rb'))])
           else:
              raise ValueError('could not find file')
        except:
           pass

        xy, yx, xy_idx, yx_idx = sess.run([net.recall_xy, net.recall_yx, net.xy_idx, net.yx_idx], feed_dict={
            net.x_data: x_batch,
            net.y_data: y_batch,
            net.K: int(1), #int(FLAGS.top_K),
            net.aff_xy: np.zeros((len(x_batch), len(y_batch))), #aff_xy,
            net.keep_prob: 1., net.is_training: False})

        print("[iter %d] xy: %s, yx: %s, " % (step, xy, yx))
        
        match = fname.split('_')[0] == str(name_batch[yx_idx[0]][0]).split('_')[0]

        if match:
           print('Best Music file: %s matches!' % str(name_batch[yx_idx[0]][0]))
        #else:
        #   print('Best Music file: %s' % str(name_batch[yx_idx[0]][0]))
  
  if run_mode == 'debug2':

     xy, yx, xy_idx, yx_idx = sess.run([net.recall_xy, net.recall_yx, net.xy_idx, net.yx_idx], feed_dict={
         net.x_data: x_batch,
         net.y_data: y_batch,
         net.K: int(FLAGS.top_K),
         net.aff_xy: aff_xy,
         net.keep_prob: 1., net.is_training: False})

     with open('./recall_xy.csv', 'w') as csvfile:
         writer = csv.writer(csvfile, delimiter=',')
         top = ["Query Name"] + [str(i) for i in range(0, num_test)]
         writer.writerow(top)
         for idx_input in range(len(xy_idx)):
             recall_mv = [name_batch[idx_recall]  for idx_recall in xy_idx[idx_input]]
             result = [name_batch[idx_input]] + recall_mv
             writer.writerow(result)

     with open('./recall_yx.csv', 'w') as csvfile:
         writer = csv.writer(csvfile, delimiter=',')
         top = ["Query Name"] + [str(i) for i in range(0, num_test)]
         writer.writerow(top)
         for idx_input in range(len(yx_idx)):
             recall_mv = [name_batch[idx_recall] for idx_recall in yx_idx[idx_input]]
             result = [name_batch[idx_input]] + recall_mv
             writer.writerow(result)
     
