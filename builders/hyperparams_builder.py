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

"""Builder function to construct tf-slim arg_scope for convolution, fc ops."""
import tensorflow as tf

from object_detection.protos import hyperparams_pb2

slim = tf.contrib.slim


def build(hyperparams_config, is_training):   #this will build a nice argument scope for the tf.slim using the configeration in conv 
  """Builds tf-slim arg_scope for convolution ops based on the config.

  Returns an arg_scope to use for convolution ops containing weights
  initializer, weights regularizer, activation function, batch norm function
  and batch norm parameters based on the configuration.

  Note that if the batch_norm parameteres are not specified in the config    #which means we have to put them or it will remain default 
  (i.e. left to default) then batch norm is excluded from the arg_scope.   #in our file no batchnorm 
 
  The batch norm parameters are set for updates based on `is_training` argument
  and conv_hyperparams_config.batch_norm.train parameter. During training, they
  are updated only if batch_norm.train parameter is true. However, during eval,
  no updates are made to the batch norm variables. In both cases, their current
  values are used during forward pass.

  Args:
    hyperparams_config: hyperparams.proto object containing
      hyperparameters.
    is_training: Whether the network is in training mode.

  Returns:
    arg_scope: tf-slim arg_scope containing hyperparameters for ops.

  Raises:
    ValueError: if hyperparams_config is not of type hyperparams.Hyperparams.
  """
  if not isinstance(hyperparams_config,
                    hyperparams_pb2.Hyperparams):
    raise ValueError('hyperparams_config not of type '
                     'hyperparams_pb.Hyperparams.')

  batch_norm = None
  batch_norm_params = None
  if hyperparams_config.HasField('batch_norm'): #if there's a batch norm . Specially spatial as far as I know 
    batch_norm = slim.batch_norm
    batch_norm_params = _build_batch_norm_params(        #buidling the batch normalization paramenters , initialising 
        hyperparams_config.batch_norm, is_training)

#depth wise separable convolove will create a feature map with depth . That means each feature map will keep speratly with out adding 
# Different convolutional operations - https://www.quora.com/What-is-the-difference-between-an-usual-convolution-layer-and-a-separable-convolution-layer-used-in-depthwise-separable-DL-models
  affected_ops = [slim.conv2d, slim.separable_conv2d, slim.conv2d_transpose] #These are the main convolution operations 
  if hyperparams_config.HasField('op') and (   #Here we don't have an op because we don't derectly predict something 
      hyperparams_config.op == hyperparams_pb2.Hyperparams.FC):
    affected_ops = [slim.fully_connected]
  with slim.arg_scope(
      affected_ops,     #what are the operations we gonna use these configuarations 
      weights_regularizer=_build_regularizer(   #what kind of regularizer l1 or l2 
          hyperparams_config.regularizer),
      weights_initializer=_build_initializer(  #initializer thing 
          hyperparams_config.initializer),
      activation_fn=_build_activation_fn(hyperparams_config.activation), #important part the activation thing 
      normalizer_fn=batch_norm,
      normalizer_params=batch_norm_params) as sc:
    return sc   #return the arg_scope 
  

def _build_activation_fn(activation_fn):
  """Builds a callable activation from config.

  Args:
    activation_fn: hyperparams_pb2.Hyperparams.activation

  Returns:
    Callable activation function.

  Raises:
    ValueError: On unknown activation function.
  """
  if activation_fn == hyperparams_pb2.Hyperparams.NONE:
    return None
  if activation_fn == hyperparams_pb2.Hyperparams.RELU:
    return tf.nn.relu
  if activation_fn == hyperparams_pb2.Hyperparams.RELU_6:
    return tf.nn.relu6
  if activation_fn == hyperparams_pb2.Hyperparams.ELU
    return tf.nn.elu  #New Also should apply prelu 
  raise ValueError('Unknown activation function: {}'.format(activation_fn))


def _build_regularizer(regularizer):
  """Builds a tf-slim regularizer from config.

  Args:
    regularizer: hyperparams_pb2.Hyperparams.regularizer proto.

  Returns:
    tf-slim regularizer.

  Raises:
    ValueError: On unknown regularizer.
  """
  regularizer_oneof = regularizer.WhichOneof('regularizer_oneof')
  if  regularizer_oneof == 'l1_regularizer':
    return slim.l1_regularizer(scale=float(regularizer.l1_regularizer.weight))
  if regularizer_oneof == 'l2_regularizer':
    return slim.l2_regularizer(scale=float(regularizer.l2_regularizer.weight))
  raise ValueError('Unknown regularizer function: {}'.format(regularizer_oneof))


def _build_initializer(initializer):
  """Build a tf initializer from config.

  Args:
    initializer: hyperparams_pb2.Hyperparams.regularizer proto.

  Returns:
    tf initializer.

  Raises:
    ValueError: On unknown initializer.
  """
  initializer_oneof = initializer.WhichOneof('initializer_oneof')
  if initializer_oneof == 'truncated_normal_initializer':
    return tf.truncated_normal_initializer(
        mean=initializer.truncated_normal_initializer.mean,
        stddev=initializer.truncated_normal_initializer.stddev)
  if initializer_oneof == 'variance_scaling_initializer':
    enum_descriptor = (hyperparams_pb2.VarianceScalingInitializer.
                       DESCRIPTOR.enum_types_by_name['Mode'])
    mode = enum_descriptor.values_by_number[initializer.
                                            variance_scaling_initializer.
                                            mode].name
    return slim.variance_scaling_initializer(
        factor=initializer.variance_scaling_initializer.factor,
        mode=mode,
        uniform=initializer.variance_scaling_initializer.uniform)
  raise ValueError('Unknown initializer function: {}'.format(
      initializer_oneof))


def _build_batch_norm_params(batch_norm, is_training):
  """Build a dictionary of batch_norm params from config.

  Args:
    batch_norm: hyperparams_pb2.ConvHyperparams.batch_norm proto.
    is_training: Whether the models is in training mode.

  Returns:
    A dictionary containing batch_norm parameters.
  """

#running_mean = momentum * running_mean + (1 - momentum) * sample_mean  Here the momentumn means the decay 
#  running_var = momentum * running_var + (1 - momentum) * sample_var
  batch_norm_params = {
      'decay': batch_norm.decay, #when calcualting the mean  #initializig the slim batch norm object 
      'center': batch_norm.center, #shit beta param    #same
      'scale': batch_norm.scale, #gamma value 
      'epsilon': batch_norm.epsilon, #variance + epsiclon 
      'fused': True,
      'is_training': is_training and batch_norm.train,
  }
  return batch_norm_params
