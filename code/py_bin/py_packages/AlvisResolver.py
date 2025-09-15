# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------------------------------------------------
AlvisResolver.py
-------------------------------------------------------------------------------------------------------------------------
Created on Fri Apr 19 16:21:09 2024

@author: Andres Cremades Botella

File for the multinode in Alvis cluster. Provided by support.
"""
# -----------------------------------------------------------------------------------------------------------------------
# Import modules
# -----------------------------------------------------------------------------------------------------------------------
import tensorflow as tf

# -----------------------------------------------------------------------------------------------------------------------
# Function to connect
# -----------------------------------------------------------------------------------------------------------------------
class AlvisResolver(tf.distribute.cluster_resolver.SlurmClusterResolver):
    def _resolve_hostlist(self):
        hosts = super()._resolve_hostlist()
        def rename(host):
            group, num = host.split('-')
            return f'{group}-{int(num):02d}'
        return [rename(host) for host in hosts]