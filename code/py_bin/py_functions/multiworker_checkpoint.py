# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 11:58:30 2024

@author:  Andres Cremades Botella

Functions extracted from the tensorflow tutorials to save the checkpoint of a multiworker
https://www.tensorflow.org/tutorials/distribute/multi_worker_with_ctl
"""
# --------------------------------------------------------------------------------------------------------------------------
# Import packages
# --------------------------------------------------------------------------------------------------------------------------
import os
import tensorflow as tf

# --------------------------------------------------------------------------------------------------------------------------
# Define functions
# --------------------------------------------------------------------------------------------------------------------------
def is_chief(task_type, task_id, cluster_spec):
    # Note: there are two possible `TF_CONFIG` configurations.
    #   1) In addition to `worker` tasks, a `chief` task type is use;
    #      in this case, this function should be modified to
    #      `return task_type == 'chief'`.
    #   2) Only `worker` task type is used; in this case, worker 0 is
    #      regarded as the chief. The implementation demonstrated here
    #      is for this case.
    # For the purpose of this Colab section, the `task_type` is `None` case
    # is added because it is effectively run with only a single worker.
    return (task_type is None or task_type == 'chief' or (task_type == 'worker' and task_id == 0
                                                          and "chief" not in cluster_spec.as_dict()))

def _get_temp_dir(dirpath, task_id):
    base_dirpath = 'workertemp_' + str(task_id)
    temp_dir = os.path.join(dirpath, base_dirpath)
    tf.io.gfile.makedirs(temp_dir)
    return temp_dir

def write_filepath(filepath, task_type, task_id, cluster_spec):
    dirpath = os.path.dirname(filepath)
    base = os.path.basename(filepath)
    if not is_chief(task_type, task_id, cluster_spec):
        dirpath = _get_temp_dir(dirpath, task_id)
    return os.path.join(dirpath, base)


# --------------------------------------------------------------------------------------------------------------------------
# Add a function to remove the workers
# --------------------------------------------------------------------------------------------------------------------------
def remove_workers(filepath, task_type, task_id, cluster_spec):
    if not is_chief(task_type, task_id, cluster_spec):
        tf.io.gfile.rmtree(os.path.dirname(filepath))