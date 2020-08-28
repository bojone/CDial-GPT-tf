#! -*- coding: utf-8 -*-
# 将CDial-GPT权重转为tf版，方便后面用bert4keras加载

import numpy as np
import torch
import tensorflow as tf
import keras.backend as K

in_file = 'GPT_LCCC-base/pytorch_model.bin'
out_file = 'GPT_LCCC-base-tf/gpt_model.ckpt'
num_hidden_layers = 12

torch_weights = torch.load(in_file, map_location='cpu')
tf_weights = {}

# CDial-GPT的[CLS]是0、[PAD]是1，不符合一般习惯，所以交换一下
w = torch_weights['transformer.tokens_embed.weight'].numpy()
w = np.concatenate([w[1:2], w[:1], w[2:]], axis=0)
tf_weights['gpt/embeddings/word_embeddings'] = w

w = torch_weights['transformer.positions_embed.weight'].numpy()
tf_weights['gpt/embeddings/position_embeddings'] = w

qkv = ['query', 'key', 'value']
for i in range(num_hidden_layers):
    w = torch_weights['transformer.h.%s.attn.c_attn.weight' % i].numpy()
    ws = np.split(w, 3, axis=1)
    for k, w in zip(qkv, ws):
        name = 'gpt/transformer/layer_%s/attention/self/%s/kernel' % (i, k)
        tf_weights[name] = w
    b = torch_weights['transformer.h.%s.attn.c_attn.bias' % i].numpy()
    bs = np.split(b, 3, axis=0)
    for k, b in zip(qkv, bs):
        name = 'gpt/transformer/layer_%s/attention/self/%s/bias' % (i, k)
        tf_weights[name] = b
    w = torch_weights['transformer.h.%s.attn.c_proj.weight' % i].numpy()
    name = 'gpt/transformer/layer_%s/attention/output/dense/kernel' % i
    tf_weights[name] = w
    b = torch_weights['transformer.h.%s.attn.c_proj.bias' % i].numpy()
    name = 'gpt/transformer/layer_%s/attention/output/dense/bias' % i
    tf_weights[name] = b
    w = torch_weights['transformer.h.%s.ln_1.weight' % i].numpy()
    name = 'gpt/transformer/layer_%s/attention/output/LayerNorm/gamma' % i
    tf_weights[name] = w
    b = torch_weights['transformer.h.%s.ln_1.bias' % i].numpy()
    name = 'gpt/transformer/layer_%s/attention/output/LayerNorm/beta' % i
    tf_weights[name] = b
    w = torch_weights['transformer.h.%s.mlp.c_fc.weight' % i].numpy()
    name = 'gpt/transformer/layer_%s/intermediate/dense/kernel' % i
    tf_weights[name] = w
    b = torch_weights['transformer.h.%s.mlp.c_fc.bias' % i].numpy()
    name = 'gpt/transformer/layer_%s/intermediate/dense/bias' % i
    tf_weights[name] = b
    w = torch_weights['transformer.h.%s.mlp.c_proj.weight' % i].numpy()
    name = 'gpt/transformer/layer_%s/output/dense/kernel' % i
    tf_weights[name] = w
    b = torch_weights['transformer.h.%s.mlp.c_proj.bias' % i].numpy()
    name = 'gpt/transformer/layer_%s/output/dense/bias' % i
    tf_weights[name] = b
    w = torch_weights['transformer.h.%s.ln_2.weight' % i].numpy()
    name = 'gpt/transformer/layer_%s/output/LayerNorm/gamma' % i
    tf_weights[name] = w
    b = torch_weights['transformer.h.%s.ln_2.bias' % i].numpy()
    name = 'gpt/transformer/layer_%s/output/LayerNorm/beta' % i
    tf_weights[name] = b

with tf.Graph().as_default():
    pairs = []
    for name, value in tf_weights.items():
        var = K.variable(tf.zeros(value.shape), name=name)
        pairs.append((var, value))
    with tf.Session() as sess:
        K.batch_set_value(pairs)
        saver = tf.train.Saver()
        saver.save(sess, out_file)
