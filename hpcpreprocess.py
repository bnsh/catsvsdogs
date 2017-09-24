#! /usr/bin/python

import sys
import os
import re
import random
import json
import shutil
import cPickle
from PIL import Image
import numpy as np

def traverse(dirname, func):
	if os.path.isdir(dirname):
		children = [os.path.join(dirname, filename) for filename in os.listdir(dirname)]
		random.shuffle(children)
		for child in children:
			traverse(child, func)
	elif os.path.isfile(dirname) and re.match(r'^.*/(cat|dog)\.([0-9]+)\.jpg$', dirname):
		func(dirname)

def process(imagedir, pickledir):
	batchsz = 256
	width = 256
	height = 256
	channels = 3

	if os.path.exists(pickledir):
		shutil.rmtree(pickledir)

	os.makedirs(pickledir, mode=0775)

	imagefilenames = []
	traverse(imagedir, imagefilenames.append)
	pickleidx = 0
	for start in xrange(0, len(imagefilenames), batchsz):
		end = min(start + batchsz, len(imagefilenames))
		actualsz = end - start
		data = np.zeros((actualsz, (width*height*channels)))
		labels = []
		for idx in xrange(start, end):
			matchgroup = re.match(r'^.*/(cat|dog)\.([0-9]+)\.jpg$', imagefilenames[idx])
			assert matchgroup is not None
			animal = matchgroup.group(1)
			imgid = int(matchgroup.group(2))
			labels.append(0 if animal == "cat" else 1)
			img = Image.open(imagefilenames[idx])
			img = img.resize((256, 256), resample=Image.BICUBIC)
			npimg = np.array(img)
			npimg = npimg.transpose(2, 0, 1).reshape(width*height*channels)
			data[idx-start] = npimg
			img.close()

		hashref = {
			"data": data,
			"labels": labels
		}
		picklefn = os.path.join(pickledir, "%08d.pkl" % (pickleidx))
		pickleidx += 1
		with open(picklefn, "w") as picklefp:
			cPickle.dump(hashref, picklefp)

		sys.stderr.write("%s: %d-%d/%d: %s\n" % (imagedir, start, end, len(imagefilenames), picklefn))

def main(argv):
	jsonfn = argv[0]
	with open(jsonfn, "r") as jsonfp:
		config = json.load(jsonfp)
		basedir = config["basedir"] 

	cropped_dir = os.path.join(basedir, "images-cropped")
	pickled_dir = os.path.join(basedir, "images-pickles")

	for mode in ("train", "validation"):
		process(os.path.join(cropped_dir, mode), os.path.join(pickled_dir, mode))

if __name__ == "__main__":
	main(sys.argv[1:])
