import tensorflow as tf
from tensorflow.keras.layers import Layer, LSTM, Bidirectional, Embedding, Input, Dense, Dropout
from tensorflow.keras.initializers import glorot_uniform


class Bi_LSTM(tf.keras.models.Model):
    #vocab_size=5,embed_size=64,units=64,num_tags=32
    def __init__(self, vocab_size, embed_size, units, num_tags, *args, **kwargs):
        super(Bi_LSTM, self).__init__()
        self.num_tags = num_tags
        # self.embedding = Embedding(input_dim=vocab_size, output_dim=embed_size, input_length=5,
        #                            embeddings_initializer="uniform",
        #                            name="embeding")
        self.fwd_LSTM = LSTM(units, return_sequences=True, go_backwards=False, dropout=0.01, name="fwd-LSTM")
        self.bwd_LSTM = LSTM(units, return_sequences=True, go_backwards=True, dropout=0.01, name="bwd-LSTM")
        self.Bi_LSTM = Bidirectional(merge_mode="concat", layer=self.fwd_LSTM,
                                    backward_layer=self.bwd_LSTM, name="Bi-LSTM")
        # merge_mode的选择从维度角度是不影响输出结果的
        self.dense = Dense(num_tags, activation='relu')
        self.dropout = Dropout(0.05)
        self.dense_1 = Dense(32, activation='relu')
        self.dense_2 = Dense(16, activation='relu')
    def call(self, inputs):
        '''inputs维度：[batch_size,max_seq_length]'''
        inputs_length = tf.math.reduce_sum(tf.cast(tf.math.not_equal(inputs, 0), dtype=tf.int32), axis=-1)
        # 自动计算每个batch的seq_length，注意数据处理时pad=0
        #x = self.embedding(inputs)
        x = self.Bi_LSTM(inputs)
        x = self.dense(x)
        x = self.dropout(x)
        x = self.dense_1(x)
        logits = self.dense_2(x)
        return logits, inputs_length

    # 定义转移矩阵transition_params
    def build(self, input_shape):
        shape = tf.TensorShape([self.num_tags, self.num_tags])
        self.transition_params = self.add_weight(name='transition_params', shape=shape, initializer=glorot_uniform,
                                                 trainable=True)
        super(Bi_LSTM, self).build(input_shape)
