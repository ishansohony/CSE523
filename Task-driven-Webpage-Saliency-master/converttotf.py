#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 11:11:13 2019

@author: nodlehs
"""
import tensorflow as tf
import os

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# images and labels array as input
def convert_to(images, filename):
  num_examples = 1
  rows = images.shape[1]
  cols = images.shape[2]
  #depth = images.shape[3]

  print('Writing', filename)
  writer = tf.python_io.TFRecordWriter(filename)
  for index in range(num_examples):
    image_raw = images[index].tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(rows),
        'width': _int64_feature(cols),
        #'depth': _int64_feature(depth),
        #'label': _int64_feature(int(labels[index])),
        'image_raw': _bytes_feature(image_raw)}))
    writer.write(example.SerializeToString())

def absoluteFilePaths(directory):
   l = []
   for dirpath,_,filenames in os.walk(directory):
       for f in filenames:
           l.append(os.path.abspath(os.path.join(dirpath, f)))
   return l
file_list = absoluteFilePaths('/home/nodlehs/Dataset/')
filename_queue = tf.train.string_input_producer(file_list) #  list of files to read

reader = tf.WholeFileReader()
key, value = reader.read(filename_queue)

my_img = tf.image.decode_png(value) # use decode_png or decode_jpeg decoder based on your files.

init_op = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init_op)

# Start populating the filename queue.

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess = sess)
    namel = os.listdir('/home/nodlehs/Dataset/')
    for i in range(len(namel)): #length of your filename list
        image = my_img.eval(session = sess) #here is your image Tensor :)
        convert_to(image, '/home/nodlehs/Datasettf/' + namel[i] + '.tfrecords')

    print(image.shape)
    #Image.show(Image.fromarray(np.asarray(image)))

    coord.request_stop()
    coord.join(threads)


