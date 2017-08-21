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

"""Classification and regression loss functions for object detection.

Localization losses:
 * WeightedL2LocalizationLoss
 * WeightedSmoothL1LocalizationLoss
 * WeightedIOULocalizationLoss

Classification losses:
 * WeightedSigmoidClassificationLoss
 * WeightedSoftmaxClassificationLoss
 * BootstrappedSigmoidClassificationLoss
"""
from abc import ABCMeta
from abc import abstractmethod

import tensorflow as tf

from object_detection.core import box_list
from object_detection.core import box_list_ops
from object_detection.utils import ops

slim = tf.contrib.slim


class Loss(object):
  """Abstract base class for loss functions."""
  __metaclass__ = ABCMeta

  def __call__(self,             #This is the class we call in the ssd)metac_arch 
               prediction_tensor,
               target_tensor,
               ignore_nan_targets=False,
               scope=None,
               **params):
    """Call the loss function.

    Args:
      prediction_tensor: a tensor representing predicted quantities.
      target_tensor: a tensor representing regression or classification targets.
      ignore_nan_targets: whether to ignore nan targets in the loss computation.
        E.g. can be used if the target tensor is missing groundtruth data that
        shouldn't be factored into the loss.
      scope: Op scope name. Defaults to 'Loss' if None.
      **params: Additional keyword arguments for specific implementations of
              the Loss.

    Returns:
      loss: a tensor representing the value of the loss function.
    """
    with tf.name_scope(scope, 'Loss',
                       [prediction_tensor, target_tensor, params]) as scope:
      if ignore_nan_targets:
        target_tensor = tf.where(tf.is_nan(target_tensor),
                                 prediction_tensor,
                                 target_tensor)
      return self._compute_loss(prediction_tensor, target_tensor, **params)  #compute the loss for the positve targents 

  @abstractmethod
  def _compute_loss(self, prediction_tensor, target_tensor, **params):
    """Method to be overriden by implementations.

    Args:
      prediction_tensor: a tensor representing predicted quantities
      target_tensor: a tensor representing regression or classification targets
      **params: Additional keyword arguments for specific implementations of
              the Loss.

    Returns:
      loss: a tensor representing the value of the loss function
    """
    pass


class WeightedL2LocalizationLoss(Loss):
  """L2 localization loss function with anchorwise output support.

  Loss[b,a] = .5 * ||weights[b,a] * (prediction[b,a,:] - target[b,a,:])||^2
  """

  def __init__(self, anchorwise_output=False):
    """Constructor.

    Args:
      anchorwise_output: Outputs loss per anchor. (default False)

    """
    self._anchorwise_output = anchorwise_output

  def _compute_loss(self, prediction_tensor, target_tensor, weights):
    """Compute loss function.

    Args:
      prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        code_size] representing the (encoded) predicted locations of objects.
      target_tensor: A float tensor of shape [batch_size, num_anchors,
        code_size] representing the regression targets
      weights: a float tensor of shape [batch_size, num_anchors]

    Returns:
      loss: a (scalar) tensor representing the value of the loss function
            or a float tensor of shape [batch_size, num_anchors]
    """
    weighted_diff = (prediction_tensor - target_tensor) * tf.expand_dims(
        weights, 2)
    square_diff = 0.5 * tf.square(weighted_diff)
    if self._anchorwise_output:
      return tf.reduce_sum(square_diff, 2)
    return tf.reduce_sum(square_diff)


class WeightedSmoothL1LocalizationLoss(Loss):
  """Smooth L1 localization loss function.

  The smooth L1_loss is defined elementwise as .5 x^2 if |x|<1 and |x|-.5 
  otherwise, where x is the difference between predictions and target.

  See also Equation (3) in the Fast R-CNN paper by Ross Girshick (ICCV 2015)
  """

  def __init__(self, anchorwise_output=False):
    """Constructor.

    Args:
      anchorwise_output: Outputs loss per anchor. (default False)

    """
    self._anchorwise_output = anchorwise_output

  def _compute_loss(self, prediction_tensor, target_tensor, weights):
    """Compute loss function.

    Args:
      prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        code_size] representing the (encoded) predicted locations of objects.
      target_tensor: A float tensor of shape [batch_size, num_anchors,
        code_size] representing the regression targets
      weights: a float tensor of shape [batch_size, num_anchors]

    Returns:
      loss: a (scalar) tensor representing the value of the loss function
    """
    diff = prediction_tensor - target_tensor
    abs_diff = tf.abs(diff)
    abs_diff_lt_1 = tf.less(abs_diff, 1)
    anchorwise_smooth_l1norm = tf.reduce_sum(
        tf.where(abs_diff_lt_1, 0.5 * tf.square(abs_diff), abs_diff - 0.5),   #here they have implemeted the loss function 
        2) * weights
    if self._anchorwise_output:
      return anchorwise_smooth_l1norm
    return tf.reduce_sum(anchorwise_smooth_l1norm)


class WeightedIOULocalizationLoss(Loss):
  """IOU localization loss function.

  Sums the IOU for corresponding pairs of predicted/groundtruth boxes
  and for each pair assign a loss of 1 - IOU.  We then compute a weighted
  sum over all pairs which is returned as the total loss.
  """

  def _compute_loss(self, prediction_tensor, target_tensor, weights):
    """Compute loss function.

    Args:
      prediction_tensor: A float tensor of shape [batch_size, num_anchors, 4]
        representing the decoded predicted boxes
      target_tensor: A float tensor of shape [batch_size, num_anchors, 4]
        representing the decoded target boxes
      weights: a float tensor of shape [batch_size, num_anchors]

    Returns:
      loss: a (scalar) tensor representing the value of the loss function
    """
    predicted_boxes = box_list.BoxList(tf.reshape(prediction_tensor, [-1, 4]))
    target_boxes = box_list.BoxList(tf.reshape(target_tensor, [-1, 4]))
    per_anchor_iou_loss = 1.0 - box_list_ops.matched_iou(predicted_boxes,
                                                         target_boxes)
    return tf.reduce_sum(tf.reshape(weights, [-1]) * per_anchor_iou_loss)


class WeightedSigmoidClassificationLoss(Loss):
  """Sigmoid cross entropy classification loss function."""

  def __init__(self, anchorwise_output=False):
    """Constructor.

    Args:
      anchorwise_output: Outputs loss per anchor. (default False)

    """
    self._anchorwise_output = anchorwise_output

  def _compute_loss(self,
                    prediction_tensor,            #get the prediction 
                    target_tensor,   
                    weights,         #This is the obtained weigts 1or zero based on the similarity default box to grount truth box 
                    class_indices=None):
    """Compute loss function.

    Args:
      prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
      target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
      weights: a float tensor of shape [batch_size, num_anchors]
      class_indices: (Optional) A 1-D integer tensor of class indices.
        If provided, computes loss only for the specified class indices.

    Returns:
      loss: a (scalar) tensor representing the value of the loss function
            or a float tensor of shape [batch_size, num_anchors]
    """

#normal way of calculating z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x)) (z is the class labels , x is the prediction )
#reed tensorflow if needed     z labels ,x is score or logit
#This is good if we have probabilistic labels , not  1 or zero hard labels 

    weights = tf.expand_dims(weights, 2)
    if class_indices is not None:
      weights *= tf.reshape(
          ops.indices_to_dense_vector(class_indices,
                                      tf.shape(prediction_tensor)[2]),
          [1, 1, -1])
    per_entry_cross_ent = (tf.nn.sigmoid_cross_entropy_with_logits(
        labels=target_tensor, logits=prediction_tensor))
    if self._anchorwise_output:
      return tf.reduce_sum(per_entry_cross_ent * weights, 2)
    return tf.reduce_sum(per_entry_cross_ent * weights)


class WeightedSoftmaxClassificationLoss(Loss):
  """Softmax loss function."""

  def __init__(self, anchorwise_output=False):
    """Constructor.

    Args:
      anchorwise_output: Whether to output loss per anchor (default False)

    """
    self._anchorwise_output = anchorwise_output

  def _compute_loss(self, prediction_tensor, target_tensor, weights):
    """Compute loss function.

    Args:
      prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
      target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
      weights: a float tensor of shape [batch_size, num_anchors]

    Returns:
      loss: a (scalar) tensor representing the value of the loss function
    """
    num_classes = prediction_tensor.get_shape().as_list()[-1]
    per_row_cross_ent = (tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.reshape(target_tensor, [-1, num_classes]),
        logits=tf.reshape(prediction_tensor, [-1, num_classes])))
    if self._anchorwise_output:
      return tf.reshape(per_row_cross_ent, tf.shape(weights)) * weights
    return tf.reduce_sum(per_row_cross_ent * tf.reshape(weights, [-1]))


class BootstrappedSigmoidClassificationLoss(Loss):
  """Bootstrapped sigmoid cross entropy classification loss function.

  This loss uses a convex combination of training labels and the current model's
  predictions as training targets in the classification loss. The idea is that
  as the model improves over time, its predictions can be trusted more and we
  can use these predictions to mitigate the damage of noisy/incorrect labels,
  because incorrect labels are likely to be eventually highly inconsistent with
  other stimuli predicted to have the same label by the model.

  In "soft" bootstrapping, we use all predicted class probabilities, whereas in
  "hard" bootstrapping, we use the single class favored by the model.

  See also Training Deep Neural Networks On Noisy Labels with Bootstrapping by
  Reed et al. (ICLR 2015).
  """

  def __init__(self, alpha, bootstrap_type='soft', anchorwise_output=False):
    """Constructor.

    Args:
      alpha: a float32 scalar tensor between 0 and 1 representing interpolation
        weight
      bootstrap_type: set to either 'hard' or 'soft' (default)
      anchorwise_output: Outputs loss per anchor. (default False)

    Raises:
      ValueError: if bootstrap_type is not either 'hard' or 'soft'
    """
    if bootstrap_type != 'hard' and bootstrap_type != 'soft':
      raise ValueError('Unrecognized bootstrap_type: must be one of '
                       '\'hard\' or \'soft.\'')
    self._alpha = alpha
    self._bootstrap_type = bootstrap_type
    self._anchorwise_output = anchorwise_output

  def _compute_loss(self, prediction_tensor, target_tensor, weights):
    """Compute loss function.

    Args:
      prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
      target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
      weights: a float tensor of shape [batch_size, num_anchors]

    Returns:
      loss: a (scalar) tensor representing the value of the loss function
            or a float tensor of shape [batch_size, num_anchors]
    """
    if self._bootstrap_type == 'soft':
      bootstrap_target_tensor = self._alpha * target_tensor + (
          1.0 - self._alpha) * tf.sigmoid(prediction_tensor)
    else:
      bootstrap_target_tensor = self._alpha * target_tensor + (
          1.0 - self._alpha) * tf.cast(
              tf.sigmoid(prediction_tensor) > 0.5, tf.float32)
    per_entry_cross_ent = (tf.nn.sigmoid_cross_entropy_with_logits(
        labels=bootstrap_target_tensor, logits=prediction_tensor))
    if self._anchorwise_output:
      return tf.reduce_sum(per_entry_cross_ent * tf.expand_dims(weights, 2), 2)
    return tf.reduce_sum(per_entry_cross_ent * tf.expand_dims(weights, 2))


class HardExampleMiner(object):
  """Hard example mining for regions in a list of images.

  Implements hard example mining to select a subset of regions to be
  back-propagated. For each image, selects the regions with highest losses,
  subject to the condition that a newly selected region cannot have
  an IOU > iou_threshold with any of the previously selected regions.
  This can be achieved by re-using a greedy non-maximum suppression algorithm.
  A constraint on the number of negatives mined per positive region can also be
  enforced.

  Reference papers: "Training Region-based Object Detectors with Online
  Hard Example Mining" (CVPR 2016) by Srivastava et al., and
  "SSD: Single Shot MultiBox Detector" (ECCV 2016) by Liu et al.
  """


#Here what is happening is first we select the regions with highest losses . Then we take another subset of regions which are more alone in the sense of IOU(This is again for selecting the hardest examples because of there's some intersection means somehow localization this has done properly but no intersection means there's a wrong thing ). This is good to avoid false postive alos . 

#we also give a ration to have number of postives per nagatives is 1 :3 from the above subset 


  def __init__(self,
               num_hard_examples=64,      #we keep this no  becausue we need all the examples to be concidered 
               iou_threshold=0.7, #we take this as one it means we take all the regions from the highest loss subclass 
               loss_type='both',
               cls_loss_weight=0.05,   #Both are 1 , This is from the config cls_loss weight and loc loss form the config 
               loc_loss_weight=0.06,
               max_negatives_per_positive=None,
               min_negatives_per_image=0):
    """Constructor.

    The hard example mining implemented by this class can replicate the behavior
    in the two aforementioned papers (Srivastava et al., and Liu et al).
    To replicate the A2 paper (Srivastava et al), num_hard_examples is set
    to a fixed parameter (64 by default) and iou_threshold is set to .7 for
    running non-max-suppression the predicted boxes prior to hard mining.
    In order to replicate the SSD paper (Liu et al), num_hard_examples should
    be set to None, max_negatives_per_positive should be 3 and iou_threshold
    should be 1.0 (in order to effectively turn off NMS).

    Args:
      num_hard_examples: maximum number of hard examples to be         #This should be none in paper 
        selected per image (prior to enforcing max negative to positive ratio
        constraint).  If set to None, all examples obtained after NMS are
        considered.
      iou_threshold: minimum intersection over union for an example
        to be discarded during NMS.
      loss_type: use only classification losses ('cls', default),
        localization losses ('loc') or both losses ('both').
        In the last case, cls_loss_weight and loc_loss_weight are used to
        compute weighted sum of the two losses.
      cls_loss_weight: weight for classification loss.
      loc_loss_weight: weight for location loss.
      max_negatives_per_positive: maximum number of negatives to retain for
        each positive anchor. By default, num_negatives_per_positive is None,   #
        which means that we do not enforce a prespecified negative:positive
        ratio.  Note also that num_negatives_per_positives can be a float
        (and will be converted to be a float even if it is passed in otherwise).
      min_negatives_per_image: minimum number of negative anchors to sample for
        a given image. Setting this to a positive number allows sampling
        negatives in an image without any positive anchors and thus not biased
        towards at least one detection per image.
    """
    self._num_hard_examples = num_hard_examples
    self._iou_threshold = iou_threshold
    self._loss_type = loss_type
    self._cls_loss_weight = cls_loss_weight
    self._loc_loss_weight = loc_loss_weight
    self._max_negatives_per_positive = max_negatives_per_positive
    self._min_negatives_per_image = min_negatives_per_image
    if self._max_negatives_per_positive is not None:
      self._max_negatives_per_positive = float(self._max_negatives_per_positive)
    self._num_positives_list = None
    self._num_negatives_list = None

  def __call__(self,                           #here we create the loss 
               location_losses,
               cls_losses,
               decoded_boxlist_list,
               match_list=None):              #here we input a matcher object in oder to do the maching IOU and other stuff 
    """Computes localization and classification losses after hard mining.

    Args:
      location_losses: a float tensor of shape [num_images, num_anchors]
        representing anchorwise localization losses.
      cls_losses: a float tensor of shape [num_images, num_anchors]
        representing anchorwise classification losses.
      decoded_boxlist_list: a list of decoded BoxList representing location
        predictions for each image.
      match_list: an optional list of matcher.Match objects encoding the match
        between anchors and groundtruth boxes for each image of the batch,
        with rows of the Match objects corresponding to groundtruth boxes
        and columns corresponding to anchors.  Match objects in match_list are
        used to reference which anchors are positive, negative or ignored.  If
        self._max_negatives_per_positive exists, these are then used to enforce
        a prespecified negative to positive ratio.

    Returns:
      mined_location_loss: a float scalar with sum of localization losses from
        selected hard examples.
      mined_cls_loss: a float scalar with sum of classification losses from
        selected hard examples.
    Raises:
      ValueError: if location_losses, cls_losses and decoded_boxlist_list do
        not have compatible shapes (i.e., they must correspond to the same
        number of images).
      ValueError: if match_list is specified but its length does not match
        len(decoded_boxlist_list).
    """
#we should input the anchorvise location and classification loss 


    mined_location_losses = []  #to store location loss for each dafault box w.t.r predticted and grount truth 
    mined_cls_losses = []        #to store class losses for each default box w.t.r predticted and grount truth 
    location_losses = tf.unstack(location_losses)
    cls_losses = tf.unstack(cls_losses)
    num_images = len(decoded_boxlist_list)      #here decode box list giving the locations per a each image . There can be many location predictions
    if not match_list:            #match list can match ground truth to the anchor boxes for each image in the batch. From this we can check whether the anchor is postive (there is a box) or negative . (Any way hard examples are selected in ssd purly from the class loss)
      match_list = num_images * [None]
    if not len(location_losses) == len(decoded_boxlist_list) == len(cls_losses): #locations for predicted boxes clas losses for each boexes same
      raise ValueError('location_losses, cls_losses and decoded_boxlist_list '
                       'do not have compatible shapes.')
    if not isinstance(match_list, list):
      raise ValueError('match_list must be a list.')
    if len(match_list) != len(decoded_boxlist_list):
      raise ValueError('match_list must either be None or have '
                       'length=len(decoded_boxlist_list).')
    num_positives_list = []                           #creating two arrays for incerting both positive and negative losses  
    num_negatives_list = []                
    for ind, detection_boxlist in enumerate(decoded_boxlist_list):           #this list contain of detetion boces 
      box_locations = detection_boxlist.get()
      match = match_list[ind]         #match between groundtruth and anchors , rows groundtruth , columns anchoes for each image
      image_losses = cls_losses[ind]      #class loss for that images . There are loss for each anchors on class have many anchor losses 
      if self._loss_type == 'loc':    #this is not our loss type 
        image_losses = location_losses[ind]  #if ony
      elif self._loss_type == 'both':            #if we need both losses we add that too 
        image_losses *= self._cls_loss_weight
        image_losses += location_losses[ind] * self._loc_loss_weight
      if self._num_hard_examples is not None:
        num_hard_examples = self._num_hard_examples       #here we take 3000 


#this is importatn this is the stage we perform the selection of hard examples . From here onwards we have set of hard examples 

#here the image_losses is only the class losses acoding to the SSD paper
      else:
        num_hard_examples = detection_boxlist.num_boxes()    #we take all the boxes if nothing is given  or the top most this number 
      selected_indices = tf.image.non_max_suppression(           #selct indices iwth the non maxima supression 
          box_locations, image_losses, num_hard_examples, self._iou_threshold)  #here the threshold is one 

#above -  selects a subset of bounding boxes in descending order of score,


#again important in the below this is now from the hard examples we need to select how many are positive (maching anchors with ground truth)

      if self._max_negatives_per_positive is not None and match:  #here it finds how many negative and postive examples in the image with machining anchor boxes and the ground trth 


#matched indices are positive, unmatched indices are negative


#indexing into a large anchor collection of N anchors where M<N) which are labeled as positive/negative via a Match object (matched indices are positive, unmatched indices are negative
        (selected_indices, num_positives,  #here number of negatives are 3 times as number of positives 
         num_negatives) = self._subsample_selection_to_desired_neg_pos_ratio(  #select number of positives and negatives 
             selected_indices, match, self._max_negatives_per_positive,
             self._min_negatives_per_image)  #obtained numb of negatives postives in the given image 

#here this function for find the things with highest loss we are not using any ROI or Jacard thing we use highest class loss. Then from that inorder to know whether that anchor box is positive or negative then we use matcher object that might check the similiraity between the anchors and bx

        num_positives_list.append(num_positives)
        num_negatives_list.append(num_negatives)
      mined_location_losses.append(
          tf.reduce_sum(tf.gather(location_losses[ind], selected_indices)))      #select  highsest location losses  
      mined_cls_losses.append(
          tf.reduce_sum(tf.gather(cls_losses[ind], selected_indices)))      #select indiced highest classes lpsses 
    location_loss = tf.reduce_sum(tf.stack(mined_location_losses))    #take the mined  location losses added in mined examples  
    cls_loss = tf.reduce_sum(tf.stack(mined_cls_losses))        #stake the mined class losses and sum hem up 
    if match and self._max_negatives_per_positive:
      self._num_positives_list = num_positives_list
      self._num_negatives_list = num_negatives_list
    return (location_loss, cls_loss)   #we sending only the slected list . 

  def summarize(self):
    """Summarize the number of positives and negatives after mining."""
    if self._num_positives_list and self._num_negatives_list:
      avg_num_positives = tf.reduce_mean(tf.to_float(self._num_positives_list))
      avg_num_negatives = tf.reduce_mean(tf.to_float(self._num_negatives_list))
      tf.summary.scalar('HardExampleMiner/NumPositives', avg_num_positives)
      tf.summary.scalar('HardExampleMiner/NumNegatives', avg_num_negatives)

  def _subsample_selection_to_desired_neg_pos_ratio(self,
                                                    indices,
                                                    match,
                                                    max_negatives_per_positive,
                                                    min_negatives_per_image=0):
    """Subsample a collection of selected indices to a desired neg:pos ratio.

    This function takes a subset of M indices (indexing into a large anchor
    collection of N anchors where M<N) which are labeled as positive/negative
    via a Match object (matched indices are positive, unmatched indices
    are negative).  It returns a subset of the provided indices retaining all
    positives as well as up to the first K negatives, where:
      K=floor(num_negative_per_positive * num_positives).

    For example, if indices=[2, 4, 5, 7, 9, 10] (indexing into 12 anchors),
    with positives=[2, 5] and negatives=[4, 7, 9, 10] and
    num_negatives_per_positive=1, then the returned subset of indices
    is [2, 4, 5, 7].

    Args:
      indices: An integer tensor of shape [M] representing a collection
        of selected anchor indices
      match: A matcher.Match object encoding the match between anchors and
        groundtruth boxes for a given image, with rows of the Match objects
        corresponding to groundtruth boxes and columns corresponding to anchors.
      max_negatives_per_positive: (float) maximum number of negatives for
        each positive anchor.
      min_negatives_per_image: minimum number of negative anchors for a given
        image. Allow sampling negatives in image without any positive anchors.

    Returns:
      selected_indices: An integer tensor of shape [M'] representing a
        collection of selected anchor indices with M' <= M.
      num_positives: An integer tensor representing the number of positive
        examples in selected set of indices.
      num_negatives: An integer tensor representing the number of negative
        examples in selected set of indices.
    """
    positives_indicator = tf.gather(match.matched_column_indicator(), indices)
    negatives_indicator = tf.gather(match.unmatched_column_indicator(), indices)
    num_positives = tf.reduce_sum(tf.to_int32(positives_indicator))
    max_negatives = tf.maximum(min_negatives_per_image,
                               tf.to_int32(max_negatives_per_positive *
                                           tf.to_float(num_positives)))
    topk_negatives_indicator = tf.less_equal(
        tf.cumsum(tf.to_int32(negatives_indicator)), max_negatives)
    subsampled_selection_indices = tf.where(
        tf.logical_or(positives_indicator, topk_negatives_indicator))
    num_negatives = tf.size(subsampled_selection_indices) - num_positives
    return (tf.reshape(tf.gather(indices, subsampled_selection_indices), [-1]),
            num_positives, num_negatives)
