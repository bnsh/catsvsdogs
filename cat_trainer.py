#! /usr/bin/python

"""This will learn the difference between cats and dogs."""

import sys
import os
import time
import random
import json
import copy
import math
import shutil
from PIL import Image, ImageFilter
import numpy as np
import tensorflow as tf

def rrelu(training, low, high):
	midval = low + (high - low) / 2.0
	def func(input_):
		loval = tf.cond(training, true_fn=lambda: low, false_fn=lambda: midval)
		hival = tf.cond(training, true_fn=lambda: high, false_fn=lambda: midval)
		return tf.maximum(tf.random_uniform(shape=tf.shape(input_), minval=loval, maxval=hival) * input_, input_)
	return func

def random_crop(resized, minsz, maxsz):
	rsz = random.randint(minsz, maxsz)
	# Now, pick a random starting point, given that rsz.
	rxx = random.randint(0, maxsz - rsz)
	ryy = random.randint(0, maxsz - rsz)

	return resized.crop((rxx, ryy, rxx+rsz, ryy+rsz))

def random_filter(cropped):
	# Now, let's do a random filter.
	# Maybe... Between a complete blur (all uniform)
	# to 9 in the center and -1.0 all around?
	# center + 8 surround == 1.0
	# 8 surround = 1.0 - center
	# surround = (1.0 - center) / 8.0
	loval = 1.0 / 9.0
	hival = 2.0 - loval
	center = loval + random.random() * (hival - loval)
	surround = (1.0 - center) / 8.0

	sharpen_or_blur = ImageFilter.Kernel((3, 3), [ \
		surround, surround, surround, \
		surround, center, surround, \
		surround, surround, surround \
	])

	return cropped.filter(sharpen_or_blur)

def random_rotation(filtered):
	width, height = filtered.size
	assert width == height
	rsz = width
	theta = -math.pi/4.0 + random.random() * math.pi / 2.0
	# WHY DEGREES?!?!
	rotated = filtered.rotate(theta * 180.0 / math.pi, resample=Image.BICUBIC, expand=False)
	# So, now what is the portion of the rotated image that
	# is still valid?
	denominator = abs(math.sin(theta)) + abs(math.cos(theta))
	rotatedsz = int(rsz / denominator)
	return rotated.crop(( \
		(rsz - rotatedsz) / 2, \
		(rsz - rotatedsz) / 2, \
		(rsz + rotatedsz) / 2, \
		(rsz + rotatedsz) / 2 \
	))

#pylint: disable=too-many-locals,too-many-arguments
def single_epoch(sess, epoch, info, input_images_, dog_predicates_, training_, action_op, loss_op, training):
	evaluation_mode = "train" if training else "validation"
	batchsz = 64

	random.shuffle(info[evaluation_mode])
	loss = 0
	valcopy = None
	if evaluation_mode == "validation":
		valcopy = copy.deepcopy(info[evaluation_mode])
	for start in xrange(0, len(info[evaluation_mode]), batchsz):
		end = min(start + batchsz, len(info[evaluation_mode]))
		data = info[evaluation_mode][start:end]
		filenames, dog_predicates = zip(*data)
		actualsz = len(data)
		np_input_images = np.zeros((actualsz, 256, 256, 3))
		np_dog_predicates = np.array(dog_predicates).reshape(len(data), 1)
		for idx in xrange(0, actualsz):
			img = Image.open(filenames[idx])
			if training:
				cropped = random_crop(img, 192, 384)
				filtered = random_filter(cropped)
				rotated = random_rotation(filtered)
				resized = rotated.resize((256, 256), resample=Image.BICUBIC)
			else:
				resized = img.resize((256, 256), resample=Image.BICUBIC)
			np_input_images[idx] = (np.array(resized) / 255.0) * 2.0 - 1.0

		batchloss, preds = sess.run([loss_op, action_op], feed_dict={input_images_: np_input_images, dog_predicates_: np_dog_predicates, training_: training})

		if valcopy is not None:
			for idx in xrange(0, actualsz):
				filename, isdog = valcopy[idx+start]
				valcopy[idx+start] = (filename, isdog, float(preds[(idx, 0)]))
		loss += batchloss * actualsz

	loss = loss / len(info[evaluation_mode])
	if valcopy is not None:
		with open("/dogs/logs/%08d.json" % (epoch), "w") as logfp:
			json.dump(valcopy, logfp, indent=4, sort_keys=True)

	return loss

#pylint: disable=not-context-manager,too-many-locals,too-many-statements
def main():
	with tf.variable_scope("configuration"):
		training_ = tf.placeholder(name="training_predicate", shape=(), dtype=tf.bool)

	activation = rrelu(training_, 1./8., 1./3.)

	with tf.variable_scope("input"):
		input_images_ = tf.placeholder(name="images", shape=(None, None, None, 3), dtype=tf.float32)

	with tf.variable_scope("targets"):
		dog_predicates_ = tf.placeholder(name="target", shape=(None, 1), dtype=tf.float32)
	# I'm just testing a hypothesis here. What if
	# I just did something really stupid, and just made conv2d's and maxpool pairs
	# till eventually I just get a 1x1x1 layer with a sigmoid at the end?

	with tf.variable_scope("conv1"):
		conv1 = tf.layers.conv2d(input_images_, filters=64, kernel_size=(11, 11), strides=(1, 1), padding="same", activation=activation, kernel_initializer=tf.glorot_normal_initializer())
		dropped1 = tf.layers.dropout(conv1, rate=0.5, noise_shape=(1, 1, 64))
		maxpool1 = tf.layers.max_pooling2d(dropped1, pool_size=(2, 2), strides=(2, 2), padding="same")
		# Now, we should be at size 128x128x64

	with tf.variable_scope("conv2"):
		conv2 = tf.layers.conv2d(maxpool1, filters=128, kernel_size=(5, 5), strides=(1, 1), padding="same", activation=activation, kernel_initializer=tf.glorot_normal_initializer())
		dropped2 = tf.layers.dropout(conv2, rate=0.5, noise_shape=(1, 1, 128))
		maxpool2 = tf.layers.max_pooling2d(dropped2, pool_size=(2, 2), strides=(2, 2), padding="same")
		# Now, we should be at size 64x64x128

	with tf.variable_scope("conv3"):
		conv3 = tf.layers.conv2d(maxpool2, filters=256, kernel_size=(5, 5), strides=(1, 1), padding="same", activation=activation, kernel_initializer=tf.glorot_normal_initializer())
		dropped3 = tf.layers.dropout(conv3, rate=0.5, noise_shape=(1, 1, 256))
		maxpool3 = tf.layers.max_pooling2d(dropped3, pool_size=(2, 2), strides=(2, 2), padding="same")
		# Now, we should be at size 32x32x256

	with tf.variable_scope("conv4"):
		conv4 = tf.layers.conv2d(maxpool3, filters=256, kernel_size=(5, 5), strides=(1, 1), padding="same", activation=activation, kernel_initializer=tf.glorot_normal_initializer())
		dropped4 = tf.layers.dropout(conv4, rate=0.5, noise_shape=(1, 1, 256))
		maxpool4 = tf.layers.max_pooling2d(dropped4, pool_size=(2, 2), strides=(2, 2), padding="same")
		# Now, we should be at size 16x16x256

	with tf.variable_scope("conv5"):
		conv5 = tf.layers.conv2d(maxpool4, filters=256, kernel_size=(5, 5), strides=(1, 1), padding="same", activation=activation, kernel_initializer=tf.glorot_normal_initializer())
		dropped5 = tf.layers.dropout(conv5, rate=0.5, noise_shape=(1, 1, 256))
		maxpool5 = tf.layers.max_pooling2d(dropped5, pool_size=(2, 2), strides=(2, 2), padding="same")
		# Now, we should be at size 8x8x256

	with tf.variable_scope("conv6"):
		conv6 = tf.layers.conv2d(maxpool5, filters=256, kernel_size=(5, 5), strides=(1, 1), padding="same", activation=activation, kernel_initializer=tf.glorot_normal_initializer())
		dropped6 = tf.layers.dropout(conv6, rate=0.5, noise_shape=(1, 1, 256))
		maxpool6 = tf.layers.max_pooling2d(dropped6, pool_size=(2, 2), strides=(2, 2), padding="same")
		# Now, we should be at size 4x4x256

	with tf.variable_scope("conv7"):
		conv7 = tf.layers.conv2d(maxpool6, filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same", activation=activation, kernel_initializer=tf.glorot_normal_initializer())
		dropped7 = tf.layers.dropout(conv7, rate=0.5, noise_shape=(1, 1, 256))
		maxpool7 = tf.layers.max_pooling2d(dropped7, pool_size=(2, 2), strides=(2, 2), padding="same")
		# Now, we should be at size 2x2x256

	with tf.variable_scope("conv8"):
		conv8 = tf.layers.conv2d(maxpool7, filters=1024, kernel_size=(1, 1), strides=(1, 1), padding="same", activation=activation, kernel_initializer=tf.glorot_normal_initializer())
		dropped8 = tf.layers.dropout(conv8, rate=0.5, noise_shape=(1, 1, 1024))
		maxpool8 = tf.layers.max_pooling2d(dropped8, pool_size=(2, 2), strides=(2, 2), padding="same")
		# Now, we should be at size 1x1x1024

	# Normally, we'd have a fully connected (dense) layer here.
	# But. What would happen if instead, I had a bunch of 1x1x1024 convs here?
	# I think then, we'd just effectively have a dense net, with the capability of
	# working with higher resolution images.
	# But, now, we can start applying dropout!
	with tf.variable_scope("conv9"):
		conv9 = tf.layers.dropout(tf.layers.conv2d(maxpool8, filters=1024, kernel_size=(1, 1), strides=(1, 1), padding="same", activation=activation, kernel_initializer=tf.glorot_normal_initializer()), rate=0.5, training=training_)
		# Now, we should _still_ be at size 1x1x1024

	with tf.variable_scope("conv10"):
		conv10 = tf.layers.dropout(tf.layers.conv2d(conv9, filters=1024, kernel_size=(1, 1), strides=(1, 1), padding="same", activation=activation, kernel_initializer=tf.glorot_normal_initializer()), rate=0.5, training=training_)
		# Now, we should _still_ be at size 1x1x1024

	with tf.variable_scope("final"):
		final = tf.layers.conv2d(conv10, filters=1, kernel_size=(1, 1), strides=(1, 1), padding="same", activation=None, kernel_initializer=tf.glorot_normal_initializer())
		# Now, we should _still_ be at size 1x1x1 __IF__ input was 256x256.

	with tf.variable_scope("cost"):
		predictions = tf.reshape(final, shape=[-1, 1])
		loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=dog_predicates_, logits=predictions)

	with tf.variable_scope("optimizer"):
		opt = tf.train.AdamOptimizer(learning_rate=0.0001)
		grad_var = opt.compute_gradients(loss)
		clipped_grad_var = [(tf.clip_by_norm(grad, 16.0), var) for grad, var in grad_var]
		step = opt.apply_gradients(clipped_grad_var)

	with tf.variable_scope("pseudo_summaries"):
		validation_cost_ = tf.placeholder(name="validation_cost", shape=(), dtype=tf.float32)
		train_cost_ = tf.placeholder(name="train_cost", shape=(), dtype=tf.float32)
		validation_cost_scalar = tf.summary.scalar("validation_cost", validation_cost_)
		train_cost_scalar = tf.summary.scalar("train_cost", train_cost_)

	tensorboarddir = "/dogs/tensorboard"
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		if os.path.exists(tensorboarddir):
			shutil.rmtree(tensorboarddir)
		if os.path.exists("/dogs/restore/bestsofar.ckpt.meta"):
			sys.stderr.write("Restoring %s\n" % ("/dogs/restore/bestsofar.ckpt"))
			restorer = tf.train.Saver()
			restorer.restore(sess, "/dogs/restore/bestsofar.ckpt")
		writer = tf.summary.FileWriter(tensorboarddir, sess.graph)
		saver = tf.train.Saver()

		with open("/dogs/images-cropped/info.json", "r") as infofp:
			info = json.load(infofp)

		epoch = 0
		best_ce = single_epoch(sess, epoch, info, input_images_, dog_predicates_, training_, loss_op=loss, action_op=predictions, training=False)
		sys.stderr.write("Starting with best_ce=%.7f\n" % (best_ce))
		while True:
			epoch_start = time.time()
			epoch += 1
			# one epoch.
			trainloss = single_epoch(sess, epoch, info, input_images_, dog_predicates_, training_, loss_op=loss, action_op=step, training=True)
			validationloss = single_epoch(sess, epoch, info, input_images_, dog_predicates_, training_, loss_op=loss, action_op=predictions, training=False)
			tcost, vcost = sess.run([train_cost_scalar, validation_cost_scalar], feed_dict={train_cost_: trainloss, validation_cost_: validationloss})
			writer.add_summary(tcost, epoch)
			writer.add_summary(vcost, epoch)

			new_best = ""
			if best_ce is None or validationloss < best_ce:
				new_best = "New Best"
				best_ce = validationloss
				saver.save(sess, os.path.join(tensorboarddir, "bestsofar.ckpt"))

			now = time.time()
			elapsed = now-epoch_start
			sys.stderr.write("elapsed: %.2fs epoch: %d: trainloss: %.7f validationloss: %.7f %s\n" % \
				(elapsed, epoch, trainloss, validationloss, new_best))


if __name__ == "__main__":
	main()
