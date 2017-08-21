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

"""SSDFeatureExtractor for MobilenetV1 features."""

import tensorflow as tf

from object_detection.meta_architectures import ssd_meta_arch
from object_detection.models import feature_map_generators
from nets import mobilenet_v1

slim = tf.contrib.slim


class SSDMobileNetV1FeatureExtractor(ssd_meta_arch.SSDFeatureExtractor):  #this is the feature extractor class 
  """SSD Feature Extractor using MobilenetV1 features."""

  def __init__(self,
               depth_multiplier,           #this uses feature extrator convolutional hipher parameters 
               min_depth,
               conv_hyperparams,
               reuse_weights=None):
    """MobileNetV1 Feature Extractor for SSD Models.

    Args:
      depth_multiplier: float depth multiplier for feature extractor.
      min_depth: minimum feature extractor depth.
      conv_hyperparams: tf slim arg_scope for conv2d and separable_conv2d ops.
      reuse_weights: Whether to reuse variables. Default is None.
    """
    super(SSDMobileNetV1FeatureExtractor, self).__init__(
        depth_multiplier, min_depth, conv_hyperparams, reuse_weights)

  def preprocess(self, resized_inputs):
    """SSD preprocessing.

    Maps pixel values to the range [-1, 1].

    Args:
      resized_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.

    Returns:
      preprocessed_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.
    """
    return (2.0 / 255.0) * resized_inputs - 1.0           

  def extract_features(self, preprocessed_inputs):       #this will extract features from iamge w.r.t mobilenet archtecture 
    """Extract features from preprocessed inputs.

    Args:
      preprocessed_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.

    Returns:
      feature_maps: a list of tensors where the ith tensor has shape
        [batch, height_i, width_i, depth_i]
    """
    preprocessed_inputs.get_shape().assert_has_rank(4)
    shape_assert = tf.Assert(
        tf.logical_and(tf.greater_equal(tf.shape(preprocessed_inputs)[1], 33),
                       tf.greater_equal(tf.shape(preprocessed_inputs)[2], 33)),
        ['image size must at least be 33 in both height and width.'])

    feature_map_layout = {
        'from_layer': ['Conv2d_11_pointwise', 'Conv2d_13_pointwise', '', '',        #we first extract 2 layers from mobilenet 
                       '', ''],
        'layer_depth': [-1, -1, 512, 256, 256, 128],       #for first two things it's -1 means we directly take the depth as in the feature maps 
    }

    with tf.control_dependencies([shape_assert]):
#with following we apply all the hyperparams in the scrip by keeping arg scope free
      with slim.arg_scope(self._conv_hyperparams): #arg score  - Here the convolutional hyper params are for feature extractor we create ot 
        with tf.variable_scope('MobilenetV1',
                               reuse=self._reuse_weights) as scope:
          _, image_features = mobilenet_v1.mobilenet_v1_base(            #getting the feature extracted from mobilnet in the slim 
              preprocessed_inputs,
              final_endpoint='Conv2d_13_pointwise',          #this is extracting the features 
              min_depth=self._min_depth,   #our min deph is 16 , It's like our depth of the feature extator 
              depth_multiplier=self._depth_multiplier, #there is 1 we take all the layers in depth demension 
              scope=scope)  #this is a dicrionalt with names of the feature maps and feature maps 


#the following function can extract the features from above feature maps , also it can create new one's too acording to the output stride thing which we are not using  Alos we give a featue map lay_out what should be there , and this also can create addicitonal feature maps 

          feature_maps = feature_map_generators.multi_resolution_feature_maps(  #This is for generating feature maps 
              feature_map_layout=feature_map_layout,  #wanted feature maps extracted from above model maps and create new maps for empty things
              depth_multiplier=self._depth_multiplier,  #depth multi-plier 
              min_depth=self._min_depth,       #this is 16 
              insert_1x1_conv=True,     # 
              image_features=image_features)  #feature dictionary 

    return feature_maps.values()     #list of 6 feature maps for the ssd 
