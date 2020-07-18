#endoced_output: Tensor("nmt/encoder/layer_preprocess/layer_norm/add_1:0", shape=(1, 6, 1024), dtype=float32)
#output Tensor("nmt/while/Exit_3:0", shape=(?, 25), dtype=int64)
#k_encdec_tensor: Tensor("nmt/body/decoder/layer_0/encdec_attention/multihead_attention/strided_slice:0", shape=(1, 6, 1024), dtype=float32)
#v_encdec_tensor: Tensor("nmt/body/decoder/layer_0/encdec_attention/multihead_attention/strided_slice_1:0", shape=(1, 6, 1024), dtype=float32)

import tensorflow as tf
import numpy as np
from tensorflow.python import pywrap_tensorflow

# model_dir = './checkpoint/'
# ckpt = tf.train.get_checkpoint_state(model_dir)
# ckpt_path = ckpt.model_checkpoint_path
# reader = pywrap_tensorflow.NewCheckpointReader(ckpt_path)
# param_dict = reader.get_variable_to_shape_map()
# for tmp in param_dict:
#     if "encoder" in tmp:
#         print(tmp)

#
ckpt = tf.train.get_checkpoint_state('./checkpoint/')  # 通过检查点文件锁定最新的模型
saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')  # 载入图结构，保存在.meta文件中
# # # sess = tf.Session()
# # # saver.restore(sess, ckpt.model_checkpoint_path)
#
#
#
#
# # #############################################self_attention###########################################
emb = tf.get_default_graph().get_tensor_by_name("nmt/bottom/encoder/embedding_shard_0:0")
input_0 = tf.nn.embedding_lookup(emb, [[12, 34, 56, 67, 78, 45]])

bias = tf.get_default_graph().get_tensor_by_name('nmt/encoder/layer_0/self_attention/layer_preprocess/layer_norm/layer_norm_bias:0')
scale = tf.get_default_graph().get_tensor_by_name('nmt/encoder/layer_0/self_attention/layer_preprocess/layer_norm/layer_norm_scale:0')
mean = tf.reduce_mean(input_0, axis=[-1], keepdims=True)
variance = tf.reduce_mean(tf.square(input_0 - mean), axis=[-1], keepdims=True)
norm_x = (input_0 - mean) * tf.rsqrt(variance + tf.constant(1e-6))
input_1 = norm_x * scale + bias


weight_1 = tf.get_default_graph().get_tensor_by_name("nmt/encoder/layer_0/self_attention/multihead_attention/q/weight:0")
q = tf.tensordot(input_1, weight_1, [[-1], [0]])
weight_2 = tf.get_default_graph().get_tensor_by_name("nmt/encoder/layer_0/self_attention/multihead_attention/k/weight:0")
k = tf.tensordot(input_1, weight_2, [[-1], [0]])
weight_3 = tf.get_default_graph().get_tensor_by_name("nmt/encoder/layer_0/self_attention/multihead_attention/v/weight:0")
v = tf.tensordot(input_1, weight_3, [[-1], [0]])
#split head
batch = 1
length = 6
q = tf.reshape(q, tf.concat([[batch, length], [16, 64]], 0))
q = tf.transpose(q, [0, 2, 1, 3])
k = tf.reshape(k, tf.concat([[batch, length], [16, 64]], 0))
k = tf.transpose(k, [0, 2, 1, 3])
v = tf.reshape(v, tf.concat([[batch, length], [16, 64]], 0))
v = tf.transpose(v, [0, 2, 1, 3])

tem_q_k = tf.matmul(q, k, transpose_b=True)
weight_v = tf.nn.softmax(tem_q_k)
input_2 = tf.matmul(weight_v, v)
x = tf.transpose(input_2, [0, 2, 1, 3])
x = tf.reshape(x, [batch, length] + [1024])

weight_last = tf.get_default_graph().get_tensor_by_name("nmt/encoder/layer_0/self_attention/multihead_attention/output_transform/weight:0")
input_3 = tf.tensordot(x, weight_last, [[-1], [0]])
output = input_0 + input_3
# # ########################################fnn_result#####################################################
bias = tf.get_default_graph().get_tensor_by_name('nmt/encoder/layer_0/ffn/layer_preprocess/layer_norm/layer_norm_bias:0')
scale = tf.get_default_graph().get_tensor_by_name('nmt/encoder/layer_0/ffn/layer_preprocess/layer_norm/layer_norm_scale:0')
mean = tf.reduce_mean(output, axis=[-1], keepdims=True)
variance = tf.reduce_mean(tf.square(output - mean), axis=[-1], keepdims=True)
norm_x = (output - mean) * tf.rsqrt(variance + tf.constant(1e-6))
input_5 = norm_x * scale + bias

weight_1 = tf.get_default_graph().get_tensor_by_name("nmt/encoder/layer_0/ffn/conv1/weight:0")
bias_1 = tf.get_default_graph().get_tensor_by_name("nmt/encoder/layer_0/ffn/conv1/bias:0")
ffn_1 = tf.tensordot(input_5, weight_1, [[-1], [0]])
ffn_1 = tf.nn.bias_add(ffn_1, bias_1)
ffn_1 = tf.nn.relu(ffn_1)
#
weight_2 = tf.get_default_graph().get_tensor_by_name("nmt/encoder/layer_0/ffn/conv2/weight:0")
bias_2 = tf.get_default_graph().get_tensor_by_name("nmt/encoder/layer_0/ffn/conv2/bias:0")
ffn_2 = tf.tensordot(ffn_1, weight_2, [[-1], [0]])
input_4 = tf.nn.bias_add(ffn_2, bias_2)
#
output = output + input_4

with tf.Session() as sess:
    saver.restore(sess, ckpt.model_checkpoint_path)
    result = sess.run(output)
    print(result)






#
# ##############################################################################################
#
#
# with tf.Session() as sess:
#     saver.restore(sess, ckpt.model_checkpoint_path)
#     result = sess.run(tf.get_default_graph().get_tensor_by_name("nmt/while/Exit_3:0"))
#     print(result)

