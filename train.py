import time
from typing import List, Tuple
import tensorflow as tf
from MKG_User.utility.decorator import logger
from MKG_User.utility.evaluation import TopkData
from MKG_User.common import log, topk

@logger('得到预测的需要训练集和验证集以及测试集')
def prepare_ds(train_data: List[Tuple[int, int, int, float, float, float]],
               test_data: List[Tuple[int, int, int, float, float, float]],
               eval_data: List[Tuple[int, int, int, float, float, float]],
               batch: int) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    def xy(data):
        user_ids = tf.constant([d[0] for d in data], dtype=tf.int32)
        item_ids = tf.constant([d[1] for d in data], dtype=tf.int32)
        labels = tf.constant([d[2] for d in data], dtype=tf.keras.backend.floatx())
        context = tf.constant([d[3] for d in data])
        review_entity_vec = tf.constant([d[4] for d in data])
        concept_vec = tf.constant([d[5] for d in data])
        return {'user_id': user_ids, 'item_id': item_ids}, labels, context, review_entity_vec, concept_vec
    train_ds = tf.data.Dataset.from_tensor_slices(xy(train_data)).shuffle(len(train_data)).batch(batch)
    test_ds = tf.data.Dataset.from_tensor_slices(xy(test_data)).batch(batch)
    eval_ds = tf.data.Dataset.from_tensor_slices(xy(eval_data)).batch(batch)
    return train_ds, test_ds, eval_ds
@logger('对TopK数据进行处理')
def get_score_fn(model):
    @tf.function(experimental_relax_shapes=True)
    def _fast_model(ui, label, context, review_entity_vec, concept_vec):
        # context = tf.expand_dims(context, axis=1)
        # review_entity_vec = tf.expand_dims(review_entity_vec, axis=1)
        # concept_vec = tf.expand_dims(concept_vec, axis=1)
        so = tf.squeeze(model([ui, label, context, review_entity_vec, concept_vec]))
        return so
        # return model([ui, label, context, review_entity_vec, concept_vec])
        # return tf.squeeze(model([ui, label, context, review_entity_vec, concept_vec]))
    def score_fn(ui, label, context, review_entity_vec, concept_vec):
        ui = {k: tf.constant(v, dtype=tf.int32) for k, v in ui.items()}
        context = tf.expand_dims(context, axis=1)
        review_entity_vec = tf.expand_dims(review_entity_vec, axis=1)
        concept_vec = tf.expand_dims(concept_vec, axis=1)
        # return _fast_model(ui, label, context, review_entity_vec, concept_vec)
        # return _fast_model(ui, label, context, review_entity_vec, concept_vec).numpy()
        s1 = _fast_model(ui, label, context, review_entity_vec, concept_vec).numpy()
        return s1
    return score_fn

@logger('开始训练，', ('epochs', 'batch'))
def train(model: tf.keras.Model,
          train_data: List[Tuple[int, int, int, float, float, float]],
          test_data: List[Tuple[int, int, int, float, float, float]],
          eval_data: List[Tuple[int, int, int,  float, float, float]],
          topk_data: TopkData = None, optimizer=None, epochs=30, batch=512):
    if optimizer is None:
        optimizer = tf.keras.optimizers.Adam()
    # 得到训练数据和测试数据通过预处理
    train_ds, test_ds, eval_ds = prepare_ds(train_data, test_data, eval_data, batch)
    # 评价指标的方法
    loss_mean_metric = tf.keras.metrics.Mean()
    auc_metric = tf.keras.metrics.AUC()
    precision_metric = tf.keras.metrics.Precision()
    recall_metric = tf.keras.metrics.Recall()
    kge_loss_mean_metric = tf.keras.metrics.Mean()
    loss_object = tf.keras.losses.BinaryCrossentropy()
    if topk_data:
        score_fn = get_score_fn(model)
    # 重置评价指标
    def reset_metrics():
        for metric in [loss_mean_metric, auc_metric, precision_metric, recall_metric, kge_loss_mean_metric]:
            tf.py_function(metric.reset_states, [], [])

    def update_metrics(loss, label, score, kge_loss):
        loss_mean_metric.update_state(loss)
        auc_metric.update_state(label, score)
        precision_metric.update_state(label, score)
        recall_metric.update_state(label, score)
        kge_loss_mean_metric.update_state(kge_loss)

    def get_metric_results():
        return loss_mean_metric.result(), auc_metric.result(), precision_metric.result(), recall_metric.result(), \
               kge_loss_mean_metric.result()

    @tf.function
    def train_batch(ui, label, context, review_entity_vec, concept_vec):
        with tf.GradientTape() as tape:
            score = model([ui, label, context, review_entity_vec, concept_vec], training=True)
            loss = loss_object(label, score) + sum(model.losses)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        update_metrics(loss, label, score, model.losses[-1])

    @tf.function
    def eval_batch(ui, label, context, review_entity_vec, concept_vec):
        score = model([ui, label, context, review_entity_vec, concept_vec])
        loss = loss_object(label, score) + sum(model.losses)
        update_metrics(loss, label, score, model.losses[-1])

    @tf.function
    def test_batch(ui, label, context, review_entity_vec, concept_vec):
        score = model([ui, label, context, review_entity_vec, concept_vec])
        print('score', score)
        loss = loss_object(label, score) + sum(model.losses)
        update_metrics(loss, label, score, model.losses[-1])

    for epoch in range(epochs):
        epoch_start_time = time.time()
        reset_metrics()
        for ui, label, context, review_entity_vec, concept_vec in train_ds:
            train_batch(ui, label, context, review_entity_vec, concept_vec)
        train_loss, train_auc, train_precision, train_recall, train_kge_loss = get_metric_results()
        # 验证集
        reset_metrics()
        for ui, label, context, review_entity_vec, concept_vec in eval_ds:
            eval_batch(ui, label, context, review_entity_vec, concept_vec)
        eval_loss, eval_auc, eval_precision, eval_recall, eval_kge_loss = get_metric_results()
        # 测试集
        reset_metrics()
        for ui, label, context, review_entity_vec, concept_vec in test_ds:
            test_batch(ui, label, context, review_entity_vec, concept_vec)
        test_loss, test_auc, test_precision, test_recall, test_kge_loss = get_metric_results()
        log(epoch, train_loss, train_auc,  train_precision,
            train_recall, test_loss, test_auc, test_precision, test_recall,
            eval_loss, eval_auc, eval_precision, eval_recall, eval_kge_loss)
        tf.print('train_kge_loss=', train_kge_loss, ', test_kge_loss=', test_kge_loss,
                 ', eval_kge_loss=', eval_kge_loss, sep='')
        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        if topk_data:
            topk(topk_data, score_fn)
        print('epoch_time=', time.time() - epoch_start_time, 's', sep='')
        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

