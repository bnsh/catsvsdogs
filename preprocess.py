#! /usr/bin/python

"""This will preprocess the cats data.
   First, it splits the train data into training and validation.
   validation is left unmolested. (Other than scaling and cropping.)
   train tho, is further cropped and rotated randomly.

   On second thought... That seems to eat up a lot of disk space.
   let's see if we can just do that at training time.
"""

import os
import re
import json
from PIL import Image

def traverse(filename, func):
	if os.path.isdir(filename):
		children = [os.path.join(filename, child) for child in sorted(os.listdir(filename))]
		for child in children:
			traverse(child, func)
	elif os.path.isfile(filename):
		func(filename)

def grab_images(argv):
	too_lo_aspect = []
	too_hi_aspect = []
	images = []
	def interrogate_image(filename):
		if re.match(r'^.*\.jpg$', filename):
			img = Image.open(filename)
			width, height = img.size
			aspectratio = width * 1.0 / height
			# These numbers were empirically calculated from
			# mean(aspectratio) +/- 3 sd.
			lo_aspect = 0.298447
			hi_aspect = 2.014374
			if lo_aspect <= aspectratio and aspectratio <= hi_aspect:
				images.append(filename)
			else:
				if aspectratio < lo_aspect:
					too_lo_aspect.append(filename)
				if aspectratio > hi_aspect:
					too_hi_aspect.append(filename)
			img.close()

	for filename in argv:
		traverse(filename, interrogate_image)

	# I find this ugly personally, I originally had this in main.
	data = {"train": [], "validation": [], "too_lo": too_lo_aspect, "too_hi": too_hi_aspect}
	return images, data

#pylint: disable=too-many-locals
def main(argv):
	images, data = grab_images(argv)

	intermediatesz = 384
	for filename in images:
		match = re.match(r'(.*)/(cat|dog)\.([0-9]+)\.jpg', filename)
		assert match is not None
		catdog = match.group(2)
		dogp = 1 if catdog == "dog" else 0
		digits = int(match.group(3))
		digdir = "%d" % (100 * int(digits / 100))
		# let's make the ones that end in 9 be our "validation" set.
		img = Image.open(filename)
		width, height = img.size
		size = min(width, height)
		cropped = img.crop(( \
			(width-size)/2, \
			(height-size)/2, \
			(width+size)/2, \
			(height+size)/2 \
		))
		resized = cropped.resize((intermediatesz, intermediatesz), resample=Image.BICUBIC)

		mode = "train"
		if digits % 10 == 9:
			mode = "validation"

		subdir = "/dogs/images-cropped/%s/%s" % (mode, digdir)
		if not os.path.exists(subdir):
			os.makedirs(subdir, mode=0775)
		newfilename = "%s/%s.%d.jpg" % (subdir, catdog, digits)
		resized.save(newfilename)
		data[mode].append((newfilename, dogp))
		print newfilename, cropped.size


		img.close()
	with open("/dogs/images-cropped/info.json", "w") as jsonfp:
		json.dump(data, jsonfp, indent=4, sort_keys=True)

if __name__ == "__main__":
	main(["/dogs/images/train"])
