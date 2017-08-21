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

"""Function to build box predictor from configuration."""

from object_detection.core import box_predictor
from object_detection.protos import box_predictor_pb2


def build(argscope_fn, box_predictor_config, is_training, num_classes):
  """Builds box predictor based on the configuration.

  Builds box predictor based on the configuration. See box_predictor.proto for
  configurable options. Also, see box_predictor.py for more details.

  Args:
    argscope_fn: A function that takes the following inputs:
        * hyperparams_pb2.Hyperparams proto
        * a boolean indicating if the model is in training mode.
      and returns a tf slim argscope for Conv and FC hyperparameters.
    box_predictor_config: box_predictor_pb2.BoxPredictor proto containing
      configuration.
    is_training: Whether the models is in training mode.
    num_classes: Number of classes to predict.

  Returns:
    box_predictor: box_predictor.BoxPredictor object.

  Raises:
    ValueError: On unknown box predictor.
  """
  if not isinstance(box_predictor_config, box_predictor_pb2.BoxPredictor):
    raise ValueError('box_predictor_config not of type '
                     'box_predictor_pb2.BoxPredictor.')

  box_predictor_oneof = box_predictor_config.WhichOneof('box_predictor_oneof') #not giben in the parameter 

  if  box_predictor_oneof == 'convolutional_box_predictor':   #we don't have this(faster RCNN) . I think this used for SSD 
    conv_box_predictor = box_predictor_config.convolutional_box_predictor  #get the params 
    conv_hyperparams = argscope_fn(conv_box_predictor.conv_hyperparams,  #converting the convolution hyper params needed for the additional conv
                                   is_training) #convolution and making help to make the arg scope . done with  hyperparametr_builder.build

#This is what following object can do   : Take high level image feature map as input and produce two predictions,
#(1) a tensor encoding box locations, and
#(2) a tensor encoding classes for each box.  - from the box_predictor.py script 
#These components are passed directly to loss functions


    box_predictor_object = box_predictor.ConvolutionalBoxPredictor(  #Initialize the boxpredictor class 
        is_training=is_training,
        num_classes=num_classes,
        conv_hyperparams=conv_hyperparams,   #above slim arg scope which have weight , operations etc 
        min_depth=conv_box_predictor.min_depth,
        max_depth=conv_box_predictor.max_depth,
        num_layers_before_predictor=(conv_box_predictor.        #how many layers before the predictoot 
                                     num_layers_before_predictor),
        use_dropout=conv_box_predictor.use_dropout,
        dropout_keep_prob=conv_box_predictor.dropout_keep_probability,
        kernel_size=conv_box_predictor.kernel_size,
        box_code_size=conv_box_predictor.box_code_size,  #what is this size of the encoding ??????????????
        apply_sigmoid_to_scores=conv_box_predictor.apply_sigmoid_to_scores)
    return box_predictor_object         #retunrs this  since box _predictor class has many functiosn 

  if box_predictor_oneof == 'mask_rcnn_box_predictor':  #this is what we have for faster Rcnn 
    mask_rcnn_box_predictor = box_predictor_config.mask_rcnn_box_predictor  #get the hyperparams 
    fc_hyperparams = argscope_fn(mask_rcnn_box_predictor.fc_hyperparams,  #this is basically getting fully connected params in order to predict 
                                 is_training)
    conv_hyperparams = None
    if mask_rcnn_box_predictor.HasField('conv_hyperparams'): #don't have one yet 
      conv_hyperparams = argscope_fn(mask_rcnn_box_predictor.conv_hyperparams,  #assigning hyper params 
                                     is_training)
    box_predictor_object = box_predictor.MaskRCNNBoxPredictor(    #this create a object from box_predictor class which use to initialize thing ss 
        is_training=is_training,
        num_classes=num_classes,
        fc_hyperparams=fc_hyperparams,
        use_dropout=mask_rcnn_box_predictor.use_dropout,
        dropout_keep_prob=mask_rcnn_box_predictor.dropout_keep_probability,
        box_code_size=mask_rcnn_box_predictor.box_code_size,
        conv_hyperparams=conv_hyperparams,
        predict_instance_masks=mask_rcnn_box_predictor.predict_instance_masks,
        mask_prediction_conv_depth=(mask_rcnn_box_predictor.
                                    mask_prediction_conv_depth),
        predict_keypoints=mask_rcnn_box_predictor.predict_keypoints)
    return box_predictor_object

  if box_predictor_oneof == 'rfcn_box_predictor':   #if rfcn we get thus 
    rfcn_box_predictor = box_predictor_config.rfcn_box_predictor
    conv_hyperparams = argscope_fn(rfcn_box_predictor.conv_hyperparams,       #then we get this slim.arguments 
                                   is_training)
    box_predictor_object = box_predictor.RfcnBoxPredictor(    #again this is for rfcn same as above 
        is_training=is_training,
        num_classes=num_classes,
        conv_hyperparams=conv_hyperparams,
        crop_size=[rfcn_box_predictor.crop_height,
                   rfcn_box_predictor.crop_width],
        num_spatial_bins=[rfcn_box_predictor.num_spatial_bins_height,
                          rfcn_box_predictor.num_spatial_bins_width],
        depth=rfcn_box_predictor.depth,
        box_code_size=rfcn_box_predictor.box_code_size)
    return box_predictor_object
  raise ValueError('Unknown box predictor: {}'.format(box_predictor_oneof))
