#! /usr/bin/python

"""This will tell the difference between cats and dogs."""

import sys
import os
from PIL import Image
import numpy as np
import tensorflow as tf

def center_crop(img):
	width, height = img.size
	rsz = min(width, height)
	rxx = (width - rsz) / 2
	ryy = (height - rsz) / 2

	return img.crop((rxx, ryy, rxx+rsz, ryy+rsz)).resize((256, 256), resample=Image.BICUBIC)

def traverse(dirname, func):
	if os.path.isdir(dirname):
		children = [os.path.join(dirname, child) for child in sorted(os.listdir(dirname))]
		for child in children:
			traverse(child, func)
	elif os.path.isfile(dirname):
		func(dirname)

#pylint: disable=not-context-manager,too-many-locals
def main(argv):
	assert os.path.exists("/dogs/restore/bestsofar.ckpt.meta")
	sys.stderr.write("Restoring %s\n" % ("/dogs/restore/bestsofar.ckpt"))
	restorer = tf.train.import_meta_graph("/dogs/restore/bestsofar.ckpt.meta")

	# Ugh. I forgot to save these in a collection like I normally do. So,
	# we have to get the variables we need by their tf ugly names.

	# we'll need input_images_, dog_predicates_, training_ and final
	graph = tf.get_default_graph()
	input_images_ = graph.get_tensor_by_name('input/images:0')
	training_ = graph.get_tensor_by_name('configuration/training_predicate:0')
	sigmoid_op = tf.nn.sigmoid(graph.get_tensor_by_name("final/conv2d/BiasAdd:0")) # should have named this. *sigh*

	batchsz = 64
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		restorer.restore(sess, "/dogs/restore/bestsofar.ckpt")

		position = [0]
		files = []
		input_images_np = np.zeros((batchsz, 256, 256, 3))

		with open("/tmp/dogs.csv", "w") as dogfp:
			def handle(files):
				sigmoids_np = sess.run(sigmoid_op, feed_dict={input_images_: input_images_np, \
					training_: False})
				sigmoids_np = sigmoids_np[0:position[0]].reshape((position[0])).tolist()
				assert len(sigmoids_np) == len(files)

				for filename, probability in zip(files, sigmoids_np):
					dogfp.write("%s,%.7f\n" % (filename, probability))
					sys.stderr.write("%s,%.7f\n" % (filename, probability))
				while files:
					files.pop()

			def process(imgfn):
				img = Image.open(imgfn)
				files.append(imgfn)
				idx = position[0]
				input_images_np[idx] = 2.0 * np.array(center_crop(img)) / 255.0 - 1
				position[0] += 1
				if position[0] == batchsz:
					handle(files)
					position[0] = 0

			for dirname in argv:
				traverse(dirname, process)
				if position[0] > 0:
					handle(files)

if __name__ == "__main__":
	main(sys.argv[1:])
