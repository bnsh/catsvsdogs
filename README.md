This is basically the Google Sandbox "homework"... We're choosing
	https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data

SO... preprocessing.

We'll have a 90/10 train/validation split, since we _already_ have a
test set supplied (and it's huge)
So... We'll have 2500 for validation and 22500 for training.
I wonder if 2500 is too big to fit into ram on colab. (About 468 megs)

Let's preprocess this so that we'll have the validation set as is.

But, the training set.. Let's make a bunch of 256x256 images that are
rotated and scaled in 16 random ways. Then, let's package them up as a
bunch of pickles. That way, we can load a "batch" of say size 500 images,
just dump them out to tensorflow and then go that way.

So, then what will we have? We'll have 22500 * 16 / 500 batches.

Eh. On second thought, that becomes enormous. Instead, I'm going to
do that rotation and scaling on the fly.
