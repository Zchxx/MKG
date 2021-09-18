import tensorflow as tf
from bert import modeling
import os

pathname = "chinese_L-12_H-768_A-12/bert_model.ckpt" # 模型地址
bert_config = modeling.BertConfig.from_json_file("chinese_L-12_H-768_A-12/bert_config.json")# 配置文件地址。
configsession = tf.ConfigProto()
configsession.gpu_options.allow_growth = True
sess = tf.Session(config=configsession)
input_ids = tf.placeholder(shape=[64, 128], dtype=tf.int32, name="input_ids")
input_mask = tf.placeholder(shape=[64, 128], dtype=tf.int32, name="input_mask")
segment_ids = tf.placeholder(shape=[64, 128], dtype=tf.int32, name="segment_ids")

with sess.as_default():
    model = modeling.BertModel(
        config=bert_config,
        is_training=True,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=False)
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())# 这里尤其注意，先初始化，在加载参数，否者会把bert的参数重新初始化。这里和demo1是有区别的
    saver.restore(sess, pathname)
    print(1)


#output_layer = model.get_sequence_output()# 这个获取每个token的output 输出[batch_size, seq_length, embedding_size] 如果做seq2seq 或者ner 用这个

output_layer = model.get_pooled_output() # 这个获取句子的output



def convert_single_example( max_seq_length,
                           tokenizer,text_a,text_b=None):
  tokens_a = tokenizer.tokenize(text_a)
  tokens_b = None
  if text_b:
    tokens_b = tokenizer.tokenize(text_b)# 这里主要是将中文分字
  if tokens_b:
    # 如果有第二个句子，那么两个句子的总长度要小于 max_seq_length - 3
    # 因为要为句子补上[CLS], [SEP], [SEP]
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
  else:
    # 如果只有一个句子，只用在前后加上[CLS], [SEP] 所以句子长度要小于 max_seq_length - 3
    if len(tokens_a) > max_seq_length - 2:
      tokens_a = tokens_a[0:(max_seq_length - 2)]

  # 转换成bert的输入，注意下面的type_ids 在源码中对应的是 segment_ids
  # (a) 两个句子:
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
  # (b) 单个句子:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0     0   0   0  0     0 0
  #
  # 这里 "type_ids" 主要用于区分第一个第二个句子。
  # 第一个句子为0，第二个句子是1。在预训练的时候会添加到单词的的向量中，但这个不是必须的
  # 因为[SEP] 已经区分了第一个句子和第二个句子。但type_ids 会让学习变的简单

  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)
  tokens.append("[SEP]")
  segment_ids.append(0)
  if tokens_b:
    for token in tokens_b:
      tokens.append(token)
      segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)
  input_ids = tokenizer.convert_tokens_to_ids(tokens)# 将中文转换成ids
  # 创建mask
  input_mask = [1] * len(input_ids)
  # 对于输入进行补0
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)
  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length
  return input_ids,input_mask,segment_ids # 对应的就是创建bert模型时候的input_ids,input_mask,segment_ids 参数
