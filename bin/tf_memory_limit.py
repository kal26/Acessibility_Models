import keras.backend
import tensorflow

configuration = tensorflow.ConfigProto()
configuration.gpu_options.allow_growth = True
configuration.intra_op_parallelism_threads = 1
configuration.inter_op_parallelism_threads = 1
configuration.device_count['CPU'] = 1
configuration.graph_options.optimizer_options.global_jit_level = tensorflow.OptimizerOptions.ON_1
session = tensorflow.Session(config=configuration)
keras.backend.set_session(session)
