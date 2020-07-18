import tensorflow as tf
import numpy as np
from tensorflow.python import pywrap_tensorflow

# model_dir = './check_point/'
# ckpt = tf.train.get_checkpoint_state(model_dir)
# ckpt_path = ckpt.model_checkpoint_path
# reader = pywrap_tensorflow.NewCheckpointReader(ckpt_path)
# param_dict = reader.get_variable_to_shape_map()
# for tmp in param_dict:
#         print(tmp)


graph = tf.Graph()

with graph.as_default():
    graph_def = tf.GraphDef()
    with tf.gfile.FastGFile('./model.pb', 'rb') as f:
        graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')  # Imports `graph_def` into the current default `Graph`

with tf.Session(graph=graph) as sess:
    input_word = [[115, 29, 112, 18, 17036, 0, 0, 0], [177, 6716, 7667, 9643, 8, 124, 0, 0]]
    input_tensor = sess.graph.get_tensor_by_name('inputs:0')
    max_decode_length = sess.graph.get_tensor_by_name('max_decode_length:0')
    decode_length_scale = sess.graph.get_tensor_by_name('decode_length_scale:0')
    target_lang = sess.graph.get_tensor_by_name('target_lang:0')
    output_tensor = sess.graph.get_tensor_by_name('greedy_targets:0')

    # for i in range(6):
    #     scale = sess.graph.get_tensor_by_name(
    #         'transformer/body/encoder/layer_{}/self_attention/layer_prepostprocess/layer_norm/layer_norm_scale/read:0'.format(
    #             i))
    #     res = sess.run(scale, feed_dict={input_tensor: input_word, max_decode_length: 100, decode_length_scale: 3,
    #                                      target_lang: [1, 1]})
    #     np.savetxt('./weight/layer_{}_self_scale.txt'.format(i), res.reshape(-1), fmt='%0.8f')
    #
    #     bias = sess.graph.get_tensor_by_name(
    #         'transformer/body/encoder/layer_{}/self_attention/layer_prepostprocess/layer_norm/layer_norm_bias/read:0'.format(
    #             i))
    #     res = sess.run(bias, feed_dict={input_tensor: input_word, max_decode_length: 100, decode_length_scale: 3,
    #                                     target_lang: [1, 1]})
    #     np.savetxt('./weight/layer_{}_self_bias.txt'.format(i), res.reshape(-1), fmt='%0.8f')
    #
    #     q = sess.graph.get_tensor_by_name(
    #         'transformer/body/encoder/layer_{}/self_attention/multihead_attention/q/kernel/read:0'.format(i))
    #     res = sess.run(q, feed_dict={input_tensor: input_word, max_decode_length: 100, decode_length_scale: 3,
    #                                  target_lang: [1, 1]})
    #     np.savetxt('./weight/layer_{}_self_q.txt'.format(i), res.reshape(-1), fmt='%0.8f')
    #
    #     k = sess.graph.get_tensor_by_name(
    #         'transformer/body/encoder/layer_{}/self_attention/multihead_attention/k/kernel/read:0'.format(i))
    #     res = sess.run(k, feed_dict={input_tensor: input_word, max_decode_length: 100, decode_length_scale: 3,
    #                                  target_lang: [1, 1]})
    #     np.savetxt('./weight/layer_{}_self_k.txt'.format(i), res.reshape(-1), fmt='%0.8f')
    #
    #     v = sess.graph.get_tensor_by_name(
    #         'transformer/body/encoder/layer_{}/self_attention/multihead_attention/v/kernel/read:0'.format(i))
    #     res = sess.run(v, feed_dict={input_tensor: input_word, max_decode_length: 100, decode_length_scale: 3,
    #                                  target_lang: [1, 1]})
    #     np.savetxt('./weight/layer_{}_self_v.txt'.format(i), res.reshape(-1), fmt='%0.8f')
    #
    #     position_key_embedding = sess.graph.get_tensor_by_name(
    #         'transformer/body/encoder/layer_{}/self_attention/multihead_attention/dot_product_attention_relative/relative_positions_keys/embeddings/read:0'.format(
    #             i))
    #     res = sess.run(position_key_embedding,
    #                    feed_dict={input_tensor: input_word, max_decode_length: 100, decode_length_scale: 3,
    #                               target_lang: [1, 1]})
    #     np.savetxt('./weight/layer_{}_self_position_key.txt'.format(i), res.reshape(-1), fmt='%0.8f')
    #
    #     position_val_embedding = sess.graph.get_tensor_by_name(
    #         'transformer/body/encoder/layer_{}/self_attention/multihead_attention/dot_product_attention_relative/relative_positions_values/embeddings/read:0'.format(
    #             i))
    #     res = sess.run(position_val_embedding,
    #                    feed_dict={input_tensor: input_word, max_decode_length: 100, decode_length_scale: 3,
    #                               target_lang: [1, 1]})
    #     np.savetxt('./weight/layer_{}_self_position_value.txt'.format(i), res.reshape(-1), fmt='%0.8f')
    #
    #     last = sess.graph.get_tensor_by_name(
    #         'transformer/body/encoder/layer_{}/self_attention/multihead_attention/output_transform/kernel/read:0'.format(
    #             i))
    #     res = sess.run(last, feed_dict={input_tensor: input_word, max_decode_length: 100, decode_length_scale: 3,
    #                                     target_lang: [1, 1]})
    #     np.savetxt('./weight/layer_{}_self_last.txt'.format(i), res.reshape(-1), fmt='%0.8f')
    #     ## ffn
    #     first_weight = sess.graph.get_tensor_by_name(
    #         'transformer/body/encoder/layer_{}/ffn/conv1/kernel/read:0'.format(i))
    #     res = sess.run(first_weight,
    #                    feed_dict={input_tensor: input_word, max_decode_length: 100, decode_length_scale: 3,
    #                               target_lang: [1, 1]})
    #     np.savetxt('./weight/layer_{}_ffn_first_weight.txt'.format(i), res.reshape(-1), fmt='%0.8f')
    #
    #     first_bias = sess.graph.get_tensor_by_name('transformer/body/encoder/layer_{}/ffn/conv1/bias/read:0'.format(i))
    #     res = sess.run(first_bias, feed_dict={input_tensor: input_word, max_decode_length: 100, decode_length_scale: 3,
    #                                           target_lang: [1, 1]})
    #     np.savetxt('./weight/layer_{}_ffn_first_bias.txt'.format(i), res.reshape(-1), fmt='%0.8f')
    #
    #     second_weight = sess.graph.get_tensor_by_name(
    #         'transformer/body/encoder/layer_{}/ffn/conv2/kernel/read:0'.format(i))
    #     res = sess.run(second_weight,
    #                    feed_dict={input_tensor: input_word, max_decode_length: 100, decode_length_scale: 3,
    #                               target_lang: [1, 1]})
    #     np.savetxt('./weight/layer_{}_ffn_second_weight.txt'.format(i), res.reshape(-1), fmt='%0.8f')
    #
    #     second_bias = sess.graph.get_tensor_by_name('transformer/body/encoder/layer_{}/ffn/conv2/bias/read:0'.format(i))
    #     res = sess.run(second_bias, feed_dict={input_tensor: input_word, max_decode_length: 100, decode_length_scale: 3,
    #                                            target_lang: [1, 1]})
    #     np.savetxt('./weight/layer_{}_ffn_second_bias.txt'.format(i), res.reshape(-1), fmt='%0.8f')
    #
    #     ffn_scale = sess.graph.get_tensor_by_name(
    #         'transformer/body/encoder/layer_{}/ffn/layer_prepostprocess/layer_norm/layer_norm_scale/read:0'.format(i))
    #     res = sess.run(ffn_scale, feed_dict={input_tensor: input_word, max_decode_length: 100, decode_length_scale: 3,
    #                                          target_lang: [1, 1]})
    #     np.savetxt('./weight/layer_{}_ffn_scale.txt'.format(i), res.reshape(-1), fmt='%0.8f')
    #
    #     ffn_bias = sess.graph.get_tensor_by_name(
    #         'transformer/body/encoder/layer_{}/ffn/layer_prepostprocess/layer_norm/layer_norm_bias/read:0'.format(i))
    #     res = sess.run(ffn_bias, feed_dict={input_tensor: input_word, max_decode_length: 100, decode_length_scale: 3,
    #                                         target_lang: [1, 1]})
    #     np.savetxt('./weight/layer_{}_ffn_bias.txt'.format(i), res.reshape(-1), fmt='%0.8f')

    # last_scale = sess.graph.get_tensor_by_name(
    #     'transformer/body/encoder/layer_prepostprocess/layer_norm/layer_norm_scale/read:0')
    # res = sess.run(last_scale, feed_dict={input_tensor: input_word, max_decode_length: 100, decode_length_scale: 3,
    #                                       target_lang: [1, 1]})
    # np.savetxt('./weight/scale.txt', res.reshape(-1), fmt='%0.8f')
    #
    # last_bias = sess.graph.get_tensor_by_name(
    #     'transformer/body/encoder/layer_prepostprocess/layer_norm/layer_norm_bias/read:0')
    # res = sess.run(last_bias, feed_dict={input_tensor: input_word, max_decode_length: 100, decode_length_scale: 3,
    #                                      target_lang: [1, 1]})
    # np.savetxt('./weight/bias.txt', res.reshape(-1), fmt='%0.8f')

    temp = sess.graph.get_tensor_by_name('transformer/body/parallel_0/body/encoder/layer_prepostprocess/layer_norm/add_1:0')
    res = sess.run(temp, feed_dict={input_tensor: input_word, max_decode_length: 100, decode_length_scale: 3, target_lang: [1, 1]})
    # np.savetxt('./weight/layer_0_ffn_bias.txt', res.reshape(-1), fmt='%0.8f')
    print(res.shape)
    print(res)

