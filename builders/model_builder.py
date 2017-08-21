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

"""A function to build a DetectionModel from configuration."""
from object_detection.builders import anchor_generator_builder                  
from object_detection.builders import box_coder_builder
from object_detection.builders import box_predictor_builder
from object_detection.builders import hyperparams_builder
from object_detection.builders import image_resizer_builder
from object_detection.builders import losses_builder
from object_detection.builders import matcher_builder
from object_detection.builders import post_processing_builder
from object_detection.builders import region_similarity_calculator_builder as sim_calc
from object_detection.core import box_predictor
from object_detection.meta_architectures import faster_rcnn_meta_arch
from object_detection.meta_architectures import rfcn_meta_arch
from object_detection.meta_architectures import ssd_meta_arch
from object_detection.models import faster_rcnn_inception_resnet_v2_feature_extractor as frcnn_inc_res
from object_detection.models import faster_rcnn_resnet_v1_feature_extractor as frcnn_resnet_v1
from object_detection.models.ssd_inception_v2_feature_extractor import SSDInceptionV2FeatureExtractor
from object_detection.models.ssd_mobilenet_v1_feature_extractor import SSDMobileNetV1FeatureExtractor
from object_detection.protos import model_pb2

# A map of names to SSD feature extractors.
SSD_FEATURE_EXTRACTOR_CLASS_MAP = {
    'ssd_inception_v2': SSDInceptionV2FeatureExtractor,
    'ssd_mobilenet_v1': SSDMobileNetV1FeatureExtractor,  #this is the one we are using 
}

# A map of names to Faster R-CNN feature extractors.
FASTER_RCNN_FEATURE_EXTRACTOR_CLASS_MAP = {        #which faster rcnn is should be used 
    'faster_rcnn_resnet50':
    frcnn_resnet_v1.FasterRCNNResnet50FeatureExtractor,
    'faster_rcnn_resnet101':
    frcnn_resnet_v1.FasterRCNNResnet101FeatureExtractor,   #A class map faster Rcnn used Resnet01
    'faster_rcnn_resnet152':
    frcnn_resnet_v1.FasterRCNNResnet152FeatureExtractor,
    'faster_rcnn_inception_resnet_v2':
    frcnn_inc_res.FasterRCNNInceptionResnetV2FeatureExtractor
}


def build(model_config, is_training):   #Here we feed the parameters with functool. 
  """Builds a DetectionModel based on the model config.

  Args:
    model_config: A model.proto object containing the config for the desired
      DetectionModel.
    is_training: True if this model is being built for training purposes.

  Returns:
    DetectionModel based on the config.

  Raises:
    ValueError: On invalid meta architecture or model.
  """
  if not isinstance(model_config, model_pb2.DetectionModel):
    raise ValueError('model_config not of type model_pb2.DetectionModel.')
  meta_architecture = model_config.WhichOneof('model') #what is the meta architecture we are using 
  if meta_architecture == 'ssd':
    return _build_ssd_model(model_config.ssd, is_training)  #sending the ssd model config params to the function 
  if meta_architecture == 'faster_rcnn':
    return _build_faster_rcnn_model(model_config.faster_rcnn, is_training)  #get the desired parameters from the config file 
  raise ValueError('Unknown meta architecture: {}'.format(meta_architecture))


def _build_ssd_feature_extractor(feature_extractor_config, is_training,
                                 reuse_weights=None):
  """Builds a ssd_meta_arch.SSDFeatureExtractor based on config.

  Args:
    feature_extractor_config: A SSDFeatureExtractor proto config from ssd.proto.
    is_training: True if this feature extractor is being built for training.
    reuse_weights: if the feature extractor should reuse weights.

  Returns:
    ssd_meta_arch.SSDFeatureExtractor based on config.

  Raises:
    ValueError: On invalid feature extractor type.
  """
  feature_type = feature_extractor_config.type   #ssd_mobilenet_v1 
  depth_multiplier = feature_extractor_config.depth_multiplier #If we want to reduce the depth in convolutional kernals 1 is the maximum 
  min_depth = feature_extractor_config.min_depth #This is the minimum depth of feature maps 
  conv_hyperparams = hyperparams_builder.build(  #Here we build the hyper params first then send it to the class 
      feature_extractor_config.conv_hyperparams, is_training) #Returns an arg_scope to use for convolution ops containing weights initializer, weights regularizer, activation function, batch norm function  and batch norm parameters based on the configuration.

  if feature_type not in SSD_FEATURE_EXTRACTOR_CLASS_MAP:
    raise ValueError('Unknown ssd feature_extractor: {}'.format(feature_type))

  feature_extractor_class = SSD_FEATURE_EXTRACTOR_CLASS_MAP[feature_type]       
  return feature_extractor_class(depth_multiplier, min_depth, conv_hyperparams,  #this is the class we initialize SSDMobileNetV1FeatureExtractor
                                 reuse_weights)     #initializa this class 


def _build_ssd_model(ssd_config, is_training):
  """Builds an SSD detection model based on the model config.

  Args:
    ssd_config: A ssd.proto object containing the config for the desired
      SSDMetaArch.
    is_training: True if this model is being built for training purposes.

  Returns:
    SSDMetaArch based on the config.
  Raises:
    ValueError: If ssd_config.type is not recognized (i.e. not registered in
      model_class_map).
  """
  num_classes = ssd_config.num_classes  #number of clases 

  # Feature extractor
  feature_extractor = _build_ssd_feature_extractor(ssd_config.feature_extractor,      #we use ssd_mobilenet_v1 as the feature extractor 
                                                   is_training)    #set the class in ssd_mobilenr_v1_feature_extractor amd ssd_meta+arch.py 

#when taking the regression loss we are working with some transorfmation. That means our predictors will predict 4 cordinates and those codinates should be regressed with some kind embedding which was made with ground truth boxes and default boxes , then after getting those we docode them for real images 


  box_coder = box_coder_builder.build(ssd_config.box_coder) #set en encoding w.r.t ground truth boxes and achor boxes . The output creating with this object will then regressed with the predicted onece. chenck equation 2 in the ssd paper 
  matcher = matcher_builder.build(ssd_config.matcher) #matching the predicted to ground trunth- Builds a matcher object based on the matcher config
#in obove object matching is done with default boxes and ground truth boxes , that's how xij value in the paper obtained . 

  region_similarity_calculator = sim_calc.build(         #how to calculate the similarity parameter is iou .
      ssd_config.similarity_calculator)

  ssd_box_predictor = box_predictor_builder.build(hyperparams_builder.build,    #This will take care of the convolutional kernal 
                                                  ssd_config.box_predictor,    
                                                  is_training, num_classes)  #this returns a box_predictor object 


  anchor_generator = anchor_generator_builder.build(         #pass an instance or object where we can create ancho boxes for differen featuremaps
      ssd_config.anchor_generator)

  image_resizer_fn = image_resizer_builder.build(ssd_config.image_resizer)    #this is imortatnt  we use   fixed_shape_resizer

  non_max_suppression_fn, score_conversion_fn = post_processing_builder.build(   #this is to work with NMS supression  output
      ssd_config.post_processing)     #score conversion function will convert logits to probabilities 

  (classification_loss, localization_loss, classification_weight,
   localization_weight,
   hard_example_miner) = losses_builder.build(ssd_config.loss)           #now the loss for hard examples  these outputs are objects 

  normalize_loss_by_num_matches = ssd_config.normalize_loss_by_num_matches # we devide by the matching acnhorboxes 

  return ssd_meta_arch.SSDMetaArch(        #here we initialized a object of ssd_meta_arch which will be used in trainign 
      is_training,
      anchor_generator,
      ssd_box_predictor,
      box_coder,
      feature_extractor,
      matcher,
      region_similarity_calculator,
      image_resizer_fn,
      non_max_suppression_fn,
      score_conversion_fn,
      classification_loss,
      localization_loss,
      classification_weight,
      localization_weight,
      normalize_loss_by_num_matches,
      hard_example_miner)


def _build_faster_rcnn_feature_extractor(
    feature_extractor_config, is_training, reuse_weights=None):
  """Builds a faster_rcnn_meta_arch.FasterRCNNFeatureExtractor based on config.

  Args:
    feature_extractor_config: A FasterRcnnFeatureExtractor proto config from
      faster_rcnn.proto.
    is_training: True if this feature extractor is being built for training.
    reuse_weights: if the feature extractor should reuse weights.

  Returns:
    faster_rcnn_meta_arch.FasterRCNNFeatureExtractor based on config.
build
  Raises:
    ValueError: On invalid feature extractor type.
  """
  feature_type = feature_extractor_config.type  #What is the type of the feature extractor Ex : faster_rcnn_resnet101
  first_stage_features_stride = (
      feature_extractor_config.first_stage_features_stride) #what is the stride , how many depth for the first stage, ex downsampled by 16

  if feature_type not in FASTER_RCNN_FEATURE_EXTRACTOR_CLASS_MAP:
    raise ValueError('Unknown Faster R-CNN feature_extractor: {}'.format(
        feature_type))
  feature_extractor_class = FASTER_RCNN_FEATURE_EXTRACTOR_CLASS_MAP[  #exact architecture frcnn_resnet_v1.FasterRCNNResnet101FeatureExtractor
      feature_type]                   
  return feature_extractor_class(                              #create the class instance of FasterRCNNResnet101FeatureExtractor   
      is_training, first_stage_features_stride, reuse_weights)


def _build_faster_rcnn_model(frcnn_config, is_training): #parameters are config of the faster RCNN
  """Builds a Faster R-CNN or R-FCN detection model based on the model config.

  Builds R-FCN model if the second_stage_box_predictor in the config is of type
  `rfcn_box_predictor` else builds a Faster R-CNN model.

  Args:
    frcnn_config: A faster_rcnn.proto object containing the config for the
    desired FasterRCNNMetaArch or RFCNMetaArch.
    is_training: True if this model is being built for training purposes.

  Returns:
    FasterRCNNMetaArch based on the config.
  Raises:
    ValueError: If frcnn_config.type is not recognized (i.e. not registered in
      model_class_map).

  """
#The config file consist of the the model parammeters 



  num_classes = frcnn_config.num_classes  #getting the classes 
  image_resizer_fn = image_resizer_builder.build(frcnn_config.image_resizer) #returns - image_resizer_fn: Callable for image resizing

  feature_extractor = _build_faster_rcnn_feature_extractor(  #create the feature extractor 
      frcnn_config.feature_extractor, is_training)      #this will take the part of the resnet as a feature extrator 

  first_stage_only = frcnn_config.first_stage_only #No field in faser Rcnn config file Since this is fale this is comple faster Rcnn
  first_stage_anchor_generator = anchor_generator_builder.build(     #here the anchor generator model preparation 
      frcnn_config.first_stage_anchor_generator)  #here inside the model we get the first_stage_anchor_generator and with that we go to the it's params 


#In above 3 outputs we get 3 functions capable of doing aboive tasks !


  first_stage_atrous_rate = frcnn_config.first_stage_atrous_rate  #not in the config file 

  first_stage_box_predictor_arg_scope = hyperparams_builder.build(    #hyper parameters builder for first stage  rpn network 
      frcnn_config.first_stage_box_predictor_conv_hyperparams, is_training) 

  first_stage_box_predictor_kernel_size = (   #This predicts the first stage conv window on the feature map of RON 
      frcnn_config.first_stage_box_predictor_kernel_size)  #not given 


  first_stage_box_predictor_depth = frcnn_config.first_stage_box_predictor_depth  #not given #Output depth for the convolution op just prior to RPN box predictions
  first_stage_minibatch_size = frcnn_config.first_stage_minibatch_size # What is the bathc size 

  first_stage_positive_balance_fraction = (                 #balance of the positive examples any way not given 
      frcnn_config.first_stage_positive_balance_fraction)

#good link to learn what is NMS - https://stackoverflow.com/questions/26691332/non-maximum-suppression-on-detected-windows-matlab

"""
1.cluster all bounding box that have overlap with each other greater than 0.5
2.for each cluster calculate the mean bounding box and output it (that is calculate the mean point between all top right corners and all bottom-right corners)

We can do this with score also 
"""
  first_stage_nms_score_threshold = frcnn_config.first_stage_nms_score_threshold  #they gave it as zero 

  first_stage_nms_iou_threshold = frcnn_config.first_stage_nms_iou_threshold #gave it 0.7
  first_stage_max_proposals = frcnn_config.first_stage_max_proposals #how many proposals in the first stage 
  first_stage_loc_loss_weight = (
      frcnn_config.first_stage_localization_loss_weight)  #This is the weight param related to regression loss in  the rpn loss function 
  first_stage_obj_loss_weight = frcnn_config.first_stage_objectness_loss_weight #same 

  initial_crop_size = frcnn_config.initial_crop_size  #crop size ?? not sure I think the feature map size 
  maxpool_kernel_size = frcnn_config.maxpool_kernel_size #ppoling kernal not sure 
  maxpool_stride = frcnn_config.maxpool_stride  # not sure 

  second_stage_box_predictor = box_predictor_builder.build(  #This will predict the boxes 
      hyperparams_builder.build,  #argoarse function retun inoder to create the box predictort (This is after the prediction frm rpn)
      frcnn_config.second_stage_box_predictor, #variables from the config file  
      is_training=is_training,
      num_classes=num_classes)


  second_stage_batch_size = frcnn_config.second_stage_batch_size  #not given 
  second_stage_balance_fraction = frcnn_config.second_stage_balance_fraction #not given 

#here this one will output the 

  (second_stage_non_max_suppression_fn, second_stage_score_conversion_fn     #this is for post processing of real predicted bpces and stuff 
  ) = post_processing_builder.build(frcnn_config.second_stage_post_processing)   #output two funtions 
  second_stage_localization_loss_weight = (   #again for the loss function 
      frcnn_config.second_stage_localization_loss_weight)
  second_stage_classification_loss_weight = (        #again for the joint loss function 
      frcnn_config.second_stage_classification_loss_weight)

  hard_example_miner = None
  if frcnn_config.HasField('hard_example_miner'):  #not given 
    hard_example_miner = losses_builder.build_hard_example_miner(  #select a subset of regions to be back-propagated(the loss)
        frcnn_config.hard_example_miner,
        second_stage_classification_loss_weight,
        second_stage_localization_loss_weight)

  common_kwargs = {
      'is_training': is_training,
      'num_classes': num_classes,
      'image_resizer_fn': image_resizer_fn,
      'feature_extractor': feature_extractor,
      'first_stage_only': first_stage_only,
      'first_stage_anchor_generator': first_stage_anchor_generator,
      'first_stage_atrous_rate': first_stage_atrous_rate,
      'first_stage_box_predictor_arg_scope':
      first_stage_box_predictor_arg_scope,
      'first_stage_box_predictor_kernel_size':
      first_stage_box_predictor_kernel_size,
      'first_stage_box_predictor_depth': first_stage_box_predictor_depth,  #Output depth for the convolution op just prior to RPN box predictions
      'first_stage_minibatch_size': first_stage_minibatch_size,
      'first_stage_positive_balance_fraction':
      first_stage_positive_balance_fraction,
      'first_stage_nms_score_threshold': first_stage_nms_score_threshold,
      'first_stage_nms_iou_threshold': first_stage_nms_iou_threshold,
      'first_stage_max_proposals': first_stage_max_proposals,
      'first_stage_localization_loss_weight': first_stage_loc_loss_weight,
      'first_stage_objectness_loss_weight': first_stage_obj_loss_weight,
      'second_stage_batch_size': second_stage_batch_size,
      'second_stage_balance_fraction': second_stage_balance_fraction,
      'second_stage_non_max_suppression_fn':
      second_stage_non_max_suppression_fn,
      'second_stage_score_conversion_fn': second_stage_score_conversion_fn,
      'second_stage_localization_loss_weight':
      second_stage_localization_loss_weight,
      'second_stage_classification_loss_weight':
      second_stage_classification_loss_weight,
      'hard_example_miner': hard_example_miner}

  if isinstance(second_stage_box_predictor, box_predictor.RfcnBoxPredictor): #this in not our intance is MaskRCNNBoxPredictor
    return rfcn_meta_arch.RFCNMetaArch(
        second_stage_rfcn_box_predictor=second_stage_box_predictor,
        **common_kwargs)
  else:
    return faster_rcnn_meta_arch.FasterRCNNMetaArch(  #this is what we use in faster rcnn ##IN the faster Rcnn meta arch py 
        maxpool_kernel_size=maxpool_kernel_size,
        maxpool_stride=maxpool_stride,             #this class object will later used to create the model 
        second_stage_mask_rcnn_box_predictor=second_stage_box_predictor,
        **common_kwargs)
