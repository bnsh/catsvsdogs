# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""CatAndDognet model configuration.

References:
  Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton
  ImageNet Classification with Deep Convolutional Neural Networks
  Advances in Neural Information Processing Systems. 2012
"""

import tensorflow as tf
import model

def rrelu(low=1.0/8.0, high=1.0/3.0):
	"""This function _generates_ a rrelu node."""
	midval = low + (high - low) / 2.0
	def func(training, input_):
		"""This is the _actual_ rrelu function"""
		loval = tf.cond(training, true_fn=lambda: low, false_fn=lambda: midval)
		hival = tf.cond(training, true_fn=lambda: high, false_fn=lambda: midval)
		return tf.maximum(tf.random_uniform(shape=tf.shape(input_), minval=loval, maxval=hival) * input_, input_)
	return func

class CatAndDognetModel(model.Model):
  """CatAndDognet cnn model."""

  def __init__(self):
    super(CatAndDognetModel, self).__init__('catanddognet', 256, 64, 0.0001)

  def add_inference(self, cnn):
    # conv1
    cnn.conv(64, 7, 7, 1, 1, 'SAME', activation=rrelu())
    cnn.dropout2d()
    cnn.mpool(2, 2, 2, 2)

    # conv2
    cnn.conv(128, 5, 5, 1, 1, 'SAME', activation=rrelu())
    cnn.dropout2d()
    cnn.mpool(2, 2, 2, 2)

    # conv3
    cnn.conv(256, 5, 5, 1, 1, 'SAME', activation=rrelu())
    cnn.dropout2d()
    cnn.mpool(2, 2, 2, 2)

    # conv4
    cnn.conv(256, 5, 5, 1, 1, 'SAME', activation=rrelu())
    cnn.dropout2d()
    cnn.mpool(2, 2, 2, 2)

    # conv5
    cnn.conv(256, 5, 5, 1, 1, 'SAME', activation=rrelu())
    cnn.dropout2d()
    cnn.mpool(2, 2, 2, 2)

    # conv6
    cnn.conv(256, 5, 5, 1, 1, 'SAME', activation=rrelu())
    cnn.dropout2d()
    cnn.mpool(2, 2, 2, 2)

    # conv7
    cnn.conv(256, 3, 3, 1, 1, 'SAME', activation=rrelu())
    cnn.dropout2d()
    cnn.mpool(2, 2, 2, 2)

    # conv8
    cnn.conv(1024, 1, 1, 1, 1, 'SAME', activation=rrelu())
    cnn.dropout2d()
    cnn.mpool(2, 2, 2, 2)

    # conv9
    cnn.conv(1024, 1, 1, 1, 1, 'SAME', activation=rrelu())
    cnn.dropout2d()
    cnn.mpool(2, 2, 2, 2)

    # conv10
    cnn.conv(1024, 1, 1, 1, 1, 'SAME', activation=rrelu())
    cnn.dropout2d()
    cnn.mpool(2, 2, 2, 2)

    # final
    cnn.conv(1, 1, 1, 1, 1, 'SAME', activation=None)
    cnn.dropout2d()
    cnn.mpool(2, 2, 2, 2)
    cnn.reshape([-1, 1])

