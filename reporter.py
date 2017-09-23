#! /usr/bin/python

"""This program will take the log files generated
   by our classifier and print out the accuracies"""

import sys
import os
import re
import math
import shutil
import json

def sigmoid(logit):
	"""Sigmoid activation function"""
	return 1.0 / (1.0 + math.exp(-logit))

def report(filename):
	"""This function will take the log files generated
	   by our classifier and print out the accuracies."""
	if os.path.exists("/tmp/misfire"):
		shutil.rmtree("/tmp/misfire")
	os.makedirs("/tmp/misfire/cats", mode=0755)
	os.makedirs("/tmp/misfire/dogs", mode=0755)
	with open(filename, "r") as jsonfp:
		data = json.load(jsonfp)

	epsilon = 1e-15
	cross_entropy = 0.0
	correct = 0
	total = 0
	for img, truth, prediction in data:
		match = re.match(r'^.*/(cat|dog)\.([0-9]+)\.jpg$', img)
		if match is not None:
			imagetype = match.group(1) + "s"
			if truth == 0 and prediction < 0:
				correct += 1
			elif truth == 1 and prediction >= 0:
				correct += 1
			else:
				shutil.copy2(img, "/tmp/misfire/%s" % (imagetype))
			if truth == 0:
				cross_entropy -= math.log(epsilon + 1.0-sigmoid(prediction))
			elif truth == 1:
				cross_entropy -= math.log(epsilon + sigmoid(prediction))
			total += 1
	sys.stderr.write("%s	%.2f%%	%.7f\n" % (filename, correct*100.0 / total, cross_entropy / total))

def main(argv):
	"""This is the main program."""
	for filename in argv:
		report(filename)

if __name__ == "__main__":
	main(sys.argv[1:])
