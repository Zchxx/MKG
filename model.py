import tensorflow as tf
from MKG_User.layer import Embedding2D
from MKG_User.utility.decorator import logger
from tensorflow.keras import layers, Sequential, optimizers
from ResNet50 import resnet50
from Bi_LSTM import Bi_LSTM


@logger('Initialize the KG_User model：', ('n_entity', 'n_relation', 'hop_size', 'kg_size',
                             'dim', 'kge_weight',  'l2', 'item_update_mode'))
def KG_User_model(n_entity: int, n_relation: int, kg_set: list,
                  hop_size=3, kg_size=32, dim=16,
                  kge_weight=0.01,  l2=1e-7, item_update_mode='plus_transform', use_all_hops=True) -> tf.keras.Model:
    assert len(kg_set[0]) == hop_size and len(kg_set[0][0][0]) == kg_size
    l2 = tf.keras.regularizers.l2(l2)
    def input_shape():
        user_id = tf.keras.Input(shape=(), name='user_id', dtype=tf.int32)
        item_id = tf.keras.Input(shape=(), name='item_id', dtype=tf.int32)
        label = tf.keras.Input(shape=(), name='label', dtype=tf.float32)
        context = tf.keras.Input(shape=(1, 300, ), name='context', dtype=tf.float64)
        review_entity = tf.keras.Input(shape=(1, 300,), name='review_entity', dtype=tf.float64)
        concept = tf.keras.Input(shape=(1, 300, ), name='concept', dtype=tf.float64)
        return user_id, item_id, context, label, review_entity, concept

    user_id, item_id, context, label, review_entity, concept = input_shape()

    entity_embedding = tf.keras.layers.Embedding(n_entity, dim, embeddings_initializer='glorot_uniform', embeddings_regularizer=l2)
    relation_embedding = Embedding2D(n_relation, dim, dim, embeddings_initializer='glorot_uniform', embeddings_regularizer=l2)
    transform_matrix = tf.keras.layers.Dense(dim, use_bias=False,
                                             kernel_initializer='glorot_uniform',
                                             kernel_regularizer=l2)

    kg_sets = tf.gather(kg_set, user_id)

    def kg_expend():
        h, r, t = [], [], []
        for hop in range(hop_size):
            h.append(entity_embedding(kg_sets[:, hop, 0]))
            r.append(relation_embedding(kg_sets[:, hop, 1]))
            t.append(entity_embedding(kg_sets[:, hop, 2]))
        return h, r, t
    h, r, t = kg_expend()

    def Deep_Attention():
        item_embedding = tf.keras.layers.Embedding(1512288, dim, embeddings_initializer='glorot_uniform',
                                                     embeddings_regularizer=l2)

        i = item_embedding(item_id)
        k_list = []
        for hop in range(hop_size):
            h_expanded = tf.expand_dims(h[hop], axis=3)
            print(h_expanded)
            Rh = tf.squeeze(tf.matmul(r[hop], h_expanded), axis=3)
            print(Rh)
            v = tf.expand_dims(i, axis=2)
            print(v)
            V_t = tf.squeeze(tf.matmul(t[hop], v), axis=2)
            print(V_t)
            Rh_v = tf.squeeze(tf.matmul(Rh, v), axis=2)
            print(Rh_v)
            deep_input = tf.concat([V_t, Rh_v], axis=1)
            print('deep_input', deep_input)
            def update_item(i, k):
                if item_update_mode == 'replace':
                    i = k
                elif item_update_mode == 'plus':
                    i = i + k
                elif item_update_mode == 'replace_transform':
                    i = transform_matrix(k)
                elif item_update_mode == 'plus_transform':
                    i = transform_matrix(i + k)
                else:
                    # 设置一个异常
                    raise Exception("Unknown item updating mode: " + item_update_mode)
                return i
            def deep_similar():
                interest_model = tf.keras.Sequential()
                interest_model.add(layers.Dense(256, activation='relu', use_bias=False, input_shape=(64,)))#TODO 64深度2倍
                interest_model.add(layers.Dense(128, activation='relu', use_bias=False))
                interest_model.add(layers.Dropout(rate=0.5))
                interest_model.add(layers.BatchNormalization(momentum=0.001, epsilon=1e-5))
                interest_model.add(layers.Dense(64, activation='relu', use_bias=False))
                interest_model.add(layers.Dropout(rate=0.5))
                # interest_model.add(layers.Dense(32, activation='relu'))
                interest_model.add(layers.Dense(32, activation='softmax'))#TODO 等于深度

                interest_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
                                   loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                                   metrics=['accuracy'])
                interest_model.summary()
                deep_out = interest_model(deep_input)
                print('deep_out', deep_out)
                return deep_out
            deep_out = deep_similar()
            weight_probs_expanded = tf.expand_dims(deep_out, axis=2)
            k = tf.reduce_sum(t[hop] * weight_probs_expanded, axis=1)
            i = update_item(i, k)
            k_list.append(k)
        interest_i = sum(k_list) if use_all_hops else k_list[-1]
        return interest_i
    interest_i = Deep_Attention()

    def sum_loss():
        kge_loss = 0
        for hop in range(hop_size):
            h_expanded = tf.expand_dims(h[hop], axis=2)
            t_expanded = tf.expand_dims(t[hop], axis=3)
            hRt = tf.squeeze(h_expanded @ r[hop] @ t_expanded)
            kge_loss += tf.reduce_mean(tf.sigmoid(hRt))
        l2_loss = 0
        for hop in range(hop_size):
            l2_loss += tf.reduce_sum(tf.square(h[hop]))
            l2_loss += tf.reduce_sum(tf.square(r[hop]))
            l2_loss += tf.reduce_sum(tf.square(t[hop]))

        return kge_loss, l2_loss

    def item_model(item_id, user_id):
        #todo 1893要改, 数据集改变的时候
        item_embedding = tf.keras.layers.Embedding(1512288, dim, embeddings_initializer='glorot_uniform',
                                                   embeddings_regularizer=l2)
        i = item_embedding(item_id)
        # i = entity_embedding(item_id)
        #TODO 输入的维度根据用户的长短改变6041
        user_embedding = tf.keras.layers.Embedding(input_dim=764937, output_dim=16)
        user_info = user_embedding(user_id)
        i_u = tf.concat([i, user_info], axis=1)
        item_model = tf.keras.Sequential()
        # item_model.add(layers.Dense(516, activation='relu', use_bias=False, input_shape=(32,)))
        # item_model.add(layers.Dense(128, activation='relu', use_bias=False))
        item_model.add(layers.Dense(64, activation='relu', use_bias=False,  input_shape=(32,)))
        item_model.add(layers.Dropout(rate=0.5))
        item_model.add(layers.Dense(32, activation='relu', use_bias=False))
        item_model.add(layers.Dense(16, activation='relu', use_bias=False))
        item_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                           loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                           metrics=['accuracy'])
        item_model.summary()
        raw_user_item = item_model(i_u)
        return raw_user_item

    def comment_sa_concept(context, review_entity, concept):
        # 这里有训练情感评分的损失loss
        def sa(context):
            l_model = Bi_LSTM(
                vocab_size=5,
                embed_size=64,
                units=64,
                num_tags=32
            )
            l_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                            loss=tf.losses.categorical_crossentropy,
                            metrics=['accuracy'])
            print('context', context)
            logits, inputs_length = l_model(context)
            #sa_score = l_model.evaluate(context, label)
            #print(sa_score)
            # l_model.summary()
            print('logits', logits)
            print('inputs_length', inputs_length)
            return logits

        # 这里有训练评论的损失loss
        def gk(context, review_entity, concept):
            print(context)
            sa_score = sa(context)  # shape=(None, 3, 16)
            print('sa_score1', sa_score)
            ResNet_model = resnet50()
            ResNet_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                                 loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                                 metrics=['accuracy'])
            ResNet_model.summary()
            sa_score = tf.squeeze(sa_score, axis=1)
            print('sa_score2', sa_score)
            print('sdasdsad', concept)
            concept = tf.expand_dims(concept, axis=3)
            print('sdasdsad', concept)
            review_entity = tf.expand_dims(review_entity, axis=3)
            # 实体用户特征
            rivew_entity = ResNet_model(review_entity) * sa_score
            # rivew_entity = ResNet_model(review_entity)
            print(rivew_entity)
            # 概念的用户特征
            concept = ResNet_model(concept) * sa_score
            # concept = ResNet_model(concept)
            user_p = tf.concat([rivew_entity, concept], axis=1)
            print('user_p', user_p)
            return user_p
        user_p = gk(context, review_entity, concept)
        return user_p
    i = item_model(item_id, user_id)
    i = tf.concat([interest_i, i], axis=1)
    print('i', i)
    user_p = comment_sa_concept(context, review_entity, concept)
    print(user_p)
    print(interest_i)
    # user_p = tf.matmul(user_p, interest_i)
    print(user_p)
    score = tf.keras.layers.Activation('sigmoid', name='score')(tf.reduce_sum(i * user_p, axis=1))
    print('score', score)

    kge_loss, l2_loss = sum_loss()

    model = tf.keras.Model(inputs=[user_id, item_id, label, context, review_entity, concept], outputs=score)
    model.add_loss(l2.l2 * l2_loss)
    model.add_loss(kge_weight * -kge_loss)
    return model
if __name__ == '__main__':
    pass
