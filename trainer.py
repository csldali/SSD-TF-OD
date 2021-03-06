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

"""Detection model trainer.

This file provides a generic training method that can be used to train a
DetectionModel.
"""

import functools

import tensorflow as tf

from object_detection.builders import optimizer_builder
from object_detection.builders import preprocessor_builder
from object_detection.core import batcher
from object_detection.core import preprocessor
from object_detection.core import standard_fields as fields
from object_detection.utils import ops as util_ops
from object_detection.utils import variables_helper
from deployment import model_deploy   #in the slim 

slim = tf.contrib.slim


def _create_input_queue(batch_size_per_clone, create_tensor_dict_fn,
                        batch_queue_capacity, num_batch_queue_threads,
                        prefetch_queue_capacity, data_augmentation_options):
  """Sets up reader, prefetcher and returns input queue.

  Args:
    batch_size_per_clone: batch size to use per clone.                 #how to set up clones ????????????
    create_tensor_dict_fn: function to create tensor dictionary.
    batch_queue_capacity: maximum number of elements to store within a queue.
    num_batch_queue_threads: number of threads to use for batching.
    prefetch_queue_capacity: maximum capacity of the queue used to prefetch
                             assembled batches.
    data_augmentation_options: a list of tuples, where each tuple contains a
      data augmentation function and a dictionary containing arguments and their
      values (see preprocessor.py).

  Returns:
    input queue: a batcher.BatchQueue object holding enqueued tensor_dicts
      (which hold images, boxes and targets).  To get a batch of tensor_dicts,
      call input_queue.Dequeue().
  """
  tensor_dict = create_tensor_dict_fn()

  tensor_dict[fields.InputDataFields.image] = tf.expand_dims(  #expand images daata , acrually etract data 
      tensor_dict[fields.InputDataFields.image], 0)

  images = tensor_dict[fields.InputDataFields.image]
  float_images = tf.to_float(images)                         #not much turning the image data in to fload
  tensor_dict[fields.InputDataFields.image] = float_images  #put that in to tensor dict

  if data_augmentation_options:   #here we will pre process 
    tensor_dict = preprocessor.preprocess(tensor_dict,     #return   tensor_dict: which contains the preprocessed images, bounding boxes, etc.
                                          data_augmentation_options)

  input_queue = batcher.BatchQueue(
      tensor_dict,
      batch_size=batch_size_per_clone,
      batch_queue_capacity=batch_queue_capacity,
      num_batch_queue_threads=num_batch_queue_threads,
      prefetch_queue_capacity=prefetch_queue_capacity)
  return input_queue


def _get_inputs(input_queue, num_classes):
  """Dequeue batch and construct inputs to object detection model.

  Args:
    input_queue: BatchQueue object holding enqueued tensor_dicts.
    num_classes: Number of classes.

  Returns:
    images: a list of 3-D float tensor of images.
    locations_list: a list of tensors of shape [num_boxes, 4]
      containing the corners of the groundtruth boxes.
    classes_list: a list of padded one-hot tensors containing target classes.
    masks_list: a list of 3-D float tensors of shape [num_boxes, image_height,
      image_width] containing instance masks for objects if present in the
      input_queue. Else returns None.
  """
  read_data_list = input_queue.dequeue()
  label_id_offset = 1
  def extract_images_and_targets(read_data):     #extractiing all the informations 
    image = read_data[fields.InputDataFields.image]
    location_gt = read_data[fields.InputDataFields.groundtruth_boxes]
    classes_gt = tf.cast(read_data[fields.InputDataFields.groundtruth_classes],
                         tf.int32)
    classes_gt -= label_id_offset
    classes_gt = util_ops.padded_one_hot_encoding(indices=classes_gt,
                                                  depth=num_classes, left_pad=0)
    masks_gt = read_data.get(fields.InputDataFields.groundtruth_instance_masks)
    return image, location_gt, classes_gt, masks_gt
  return zip(*map(extract_images_and_targets, read_data_list)) #


def _create_losses(input_queue, create_model_fn):
  """Creates loss function for a DetectionModel.

  Args:
    input_queue: BatchQueue object holding enqueued tensor_dicts.
    create_model_fn: A function to create the DetectionModel.
  """
  detection_model = create_model_fn()    #this is an initialized  object of  ssd_meta_arch.SSDMetaArch

  (images, groundtruth_boxes_list, groundtruth_classes_list,
   groundtruth_masks_list
  ) = _get_inputs(input_queue, detection_model.num_classes)    #pure inputs from the 

#Here the following function will fixed the shape of the image in SSd and make pixels 0 and one

  images = [detection_model.preprocess(image) for image in images]  #preprocess images in to given aspect ratio and zero cenerd with mean pixel

  images = tf.concat(images, 0)  #concatanation of images 
  if any(mask is None for mask in groundtruth_masks_list):  #detection model comes from faster_rcnn_meta_arch.py in FasterRCNNMetaArch class
    groundtruth_masks_list = None

  detection_model.provide_groundtruth(groundtruth_boxes_list,   #Assigning the data
                                      groundtruth_classes_list,
                                      groundtruth_masks_list)
  prediction_dict = detection_model.predict(images) #run the forwad pass this fucntion is in faster_rcnn_meta_arch.py

#IN above function we create the model acording to the output stride 


#Here to calculate the loss we input box predictions and class predictions for each feature maps from above  outut 

  losses_dict = detection_model.loss(prediction_dict)  
  for loss_tensor in losses_dict.values():
    tf.losses.add_loss(loss_tensor)   #Get the loss minde regression and mined classfication loss , online hard example method 


#all the functions are in faster_rcnn_meta_arch.py . faster_rcnn_resnet_v1_feature_extrator.py , model.py 




def train(create_tensor_dict_fn, create_model_fn, train_config, master, task,
          num_clones, worker_replicas, clone_on_cpu, ps_tasks, worker_job_name,
          is_chief, train_dir):
  """Training function for detection models.

  Args:
    create_tensor_dict_fn: a function to create a tensor input dictionary.
    create_model_fn: a function that creates a DetectionModel and generates
                     losses.
    train_config: a train_pb2.TrainConfig protobuf.
    master: BNS name of the TensorFlow master to use.
    task: The task id of this training instance.
    num_clones: The number of clones to run per machine.
    worker_replicas: The number of work replicas to train with.
    clone_on_cpu: True if clones should be forced to run on CPU.
    ps_tasks: Number of parameter server tasks.
    worker_job_name: Name of the worker job.
    is_chief: Whether this replica is the chief replica.
    train_dir: Directory to write checkpoints and training summaries to.
  """

  detection_model = create_model_fn()    #Object for create the detection model 
  data_augmentation_options = [             #for ssd it's ssd random crop 
      preprocessor_builder.build(step)  #random_horizontal_flip in the faster rcnn config file 
      for step in train_config.data_augmentation_options]

  with tf.Graph().as_default():   #we need a default graph in order to create the model 
    # Build a configuration specifying multi-GPU and multi-replicas.
    deploy_config = model_deploy.DeploymentConfig(   
        num_clones=num_clones,
        clone_on_cpu=clone_on_cpu,
        replica_id=task,
        num_replicas=worker_replicas,
        num_ps_tasks=ps_tasks,
        worker_job_name=worker_job_name)

    # Place the global step on the device storing the variables.    #global step is needed to keep the records 
    with tf.device(deploy_config.variables_device()):  #suitable device for operation  +++On CPU I think 
      global_step = slim.create_global_step()  #created the global step tensor 


#The following will create an input Que images ,boxes m targets 
    with tf.device(deploy_config.inputs_device()):  #Device to use to build the inputs ++++on CPU ?? 
      input_queue = _create_input_queue(train_config.batch_size // num_clones,  #here batch size/number_clones 
                                        create_tensor_dict_fn,    
                                        train_config.batch_queue_capacity,
                                        train_config.num_batch_queue_threads,
                                        train_config.prefetch_queue_capacity,
                                        data_augmentation_options) #random_horizontal_flip 

    # Gather initial summaries.
    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))    #vreate the summeries 
    global_summaries = set([])
                                                #Creating the loss
    model_fn = functools.partial(_create_losses,             #This will create the losses , It need a object of our model as an argivement 
                                 create_model_fn=create_model_fn)
    clones = model_deploy.create_clones(deploy_config, model_fn, [input_queue])  #creating the clones with respect to t he input model fn 
    first_clone_scope = clones[0].scope

    # Gather update_ops from the first clone. These contain, for example,
    # the updates for the batch_norm variables created by model_fn.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)

    with tf.device(deploy_config.optimizer_device()):  #This is important 
      training_optimizer = optimizer_builder.build(train_config.optimizer,  #optimization 
                                                   global_summaries)    #will select rms_prop , Adam Here derectly we get the optimizer

    sync_optimizer = None
    if train_config.sync_replicas:
      training_optimizer = tf.SyncReplicasOptimizer(     #This is more of synchronising the optimizer because there are repicas doing optimizing
          training_optimizer,
          replicas_to_aggregate=train_config.replicas_to_aggregate,
          total_num_replicas=train_config.worker_replicas)
      sync_optimizer = training_optimizer

    # Create ops required to initialize the model from a given checkpoint.
    init_fn = None
    if train_config.fine_tune_checkpoint:    #This is the checkpoint path file 
      init_fn = detection_model.restore_fn(          #Re storing the weights from the feature extractors 
          train_config.fine_tune_checkpoint,
          from_detection_checkpoint=train_config.from_detection_checkpoint)      #This is more of the initializer which is re-stored from check points  

    with tf.device(deploy_config.optimizer_device()):
      total_loss, grads_and_vars = model_deploy.optimize_clones(      #This gives the total loss and also the grad and var pairs (Tuple) 
          clones, training_optimizer, regularization_losses=None)
      total_loss = tf.check_numerics(total_loss, 'LossTensor is inf or nan.')

      # Optionally multiply bias gradients by train_config.bias_grad_multiplier.
      if train_config.bias_grad_multiplier:          #We have not initialized a bias gradient multiplier 
        biases_regex_list = ['.*/biases']
        grads_and_vars = variables_helper.multiply_gradients_matching_regex(
            grads_and_vars,
            biases_regex_list,
            multiplier=train_config.bias_grad_multiplier)

      # Optionally freeze some layers by setting their gradients to be zero.
      if train_config.freeze_variables:                       #Here we are not freezing any may be it's good to freeze the
#This will be usefult to go through the variables 
        print("Priting the grad_and_vars to check the tuples ")
        print(grad_and_vars) 
        grads_and_vars = variables_helper.freeze_gradients_matching_regex(  #input to this also grads and vars which means 
            grads_and_vars, train_config.freeze_variables)     #This function will output 
                                            #We are getiing gradients and of their varaibles exept the froxen list 
      # Optionally clip gradients
      if train_config.gradient_clipping_by_norm > 0:
        with tf.name_scope('clip_grads'):
          grads_and_vars = slim.learning.clip_gradient_norms(
              grads_and_vars, train_config.gradient_clipping_by_norm)

      # Create gradient updates.
      grad_updates = training_optimizer.apply_gradients(grads_and_vars,        #updating the gradinets list 
                                                        global_step=global_step)
      update_ops.append(grad_updates) #Here the new updated variables 

      update_op = tf.group(*update_ops)
      with tf.control_dependencies([update_op]):
        train_tensor = tf.identity(total_loss, name='train_op')

    # Add summaries.
    for model_var in slim.get_model_variables():
      global_summaries.add(tf.summary.histogram(model_var.op.name, model_var))
    for loss_tensor in tf.losses.get_losses():
      global_summaries.add(tf.summary.scalar(loss_tensor.op.name, loss_tensor))
    global_summaries.add(
        tf.summary.scalar('TotalLoss', tf.losses.get_total_loss()))

    # Add the summaries from the first clone. These contain the summaries
    # created by model_fn and either optimize_clones() or _gather_clone_loss().
    summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES,
                                       first_clone_scope))
    summaries |= global_summaries

    # Merge all summaries together.
    summary_op = tf.summary.merge(list(summaries), name='summary_op')

    # Soft placement allows placing on CPU ops without GPU implementation.
    session_config = tf.ConfigProto(allow_soft_placement=True,
                                    log_device_placement=False)

    # Save checkpoints regularly.
    keep_checkpoint_every_n_hours = train_config.keep_checkpoint_every_n_hours
    saver = tf.train.Saver(                  #saving the checkpoints 
        keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours)

    slim.learning.train(             #Training the network using a compact function 
        train_tensor,
        logdir=train_dir,
        master=master,
        is_chief=is_chief,
        session_config=session_config,
        startup_delay_steps=train_config.startup_delay_steps,
        init_fn=init_fn,
        summary_op=summary_op,
        number_of_steps=(
            train_config.num_steps if train_config.num_steps else None),
        save_summaries_secs=120,
        sync_optimizer=sync_optimizer,
        saver=saver)
