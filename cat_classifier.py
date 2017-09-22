#! /usr/bin/python

"""This will tell the difference between cats and dogs."""

import sys
import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf

def center_crop(img):
	width, height = img.size
	rsz = min(width, height)
	rxx = (width - rsz) / 2
	ryy = (height - rsz) / 2

	return img.crop((rxx, ryy, rxx+rsz, ryy+rsz)).resize((256, 256), resample=Image.BICUBIC)


#pylint: disable=not-context-manager,unused-variable
def main(argv):
	tensorboarddir = "/dogs/tensorboard"
	batchsz = 64
	assert os.path.exists("/dogs/restore/bestsofar.ckpt.meta")
	sys.stderr.write("Restoring %s\n" % ("/dogs/restore/bestsofar.ckpt"))
	restorer = tf.train.import_meta_graph("/dogs/restore/bestsofar.ckpt.meta")

	# Ugh. I forgot to save these in a collection like I normally do. So,
	# we have to get the variables we need by their tf ugly names.

	# we'll need input_images_, dog_predicates_, training_ and final
	graph = tf.get_default_graph()
	input_images_ = graph.get_tensor_by_name('input/images:0')
	dog_predicates_ = graph.get_tensor_by_name('targets/target:0')
	training_ = graph.get_tensor_by_name('configuration/training_predicate:0')
	logits_op = tf.nn.sigmoid(graph.get_tensor_by_name("final/conv2d/BiasAdd:0")) # should have named this. *sigh*

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		restorer.restore(sess, "/dogs/restore/bestsofar.ckpt")

		input_images_np = np.zeros((len(argv), 256, 256, 3))
		for idx, imgfn in enumerate(argv):
			img = Image.open(imgfn)
			input_images_np[idx] = np.array(center_crop(img))
			img.close()

		logits_np = sess.run(logits_op, feed_dict={input_images_: input_images_np, \
			training_: False})
		assert len(logits_np) == len(argv)
		json.dump(dict(zip(argv, logits_np.reshape((len(argv))).tolist())), sys.stdout, indent=4, sort_keys=True)


if __name__ == "__main__":
	main(sys.argv[1:])
