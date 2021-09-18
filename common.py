from typing import List, Callable, Dict
from MKG_User.utility.evaluation import TopkData, topk_evaluate
from MKG_User.utility.decorator import logger

def log(epoch, train_loss, train_auc, train_precision, train_recall, test_loss, test_auc,  test_precision, test_recall,
        eval_loss, eval_auc, eval_precision, eval_recall, eval_kge_loss):
    train_f1 = 2. * train_precision * train_recall / (train_precision + train_recall) if train_precision + train_recall else 0
    eval_f1 = 2. * eval_precision * eval_recall / (eval_precision + eval_recall) if eval_precision + eval_recall else 0
    test_f1 = 2. * test_precision * test_recall / (test_precision + test_recall) if test_precision + test_recall else 0

    print('epoch=%d: \n '
          'train_loss=%.5f, train_auc=%.5f,  train_precision=%.5f, train_f1=%.5f,\n '
          'eval_loss=%.5f, eval_auc=%.5f,  eval_precision=%.5f, eval_f1=%.5f\n '
          'test_loss=%.5f, test_auc=%.5f,  test_precision=%.5f, test_f1=%.5f '
          % (epoch + 1,
           train_loss, train_auc,  train_precision, train_f1,
           eval_loss, eval_auc, eval_precision, eval_f1,
           test_loss, test_auc, test_precision, test_f1))
from typing import List, Tuple
@logger('处理topK的数据')
# todo def topk(topk_data: TopkData, score_fn: Callable[[Dict[str, List[int]]], List[float]], ks=[10, 36, 100]):
def topk(topk_data: TopkData, score_fn):
    ks = [10, 36, 64, 100]
    precisions, recalls = topk_evaluate(topk_data, score_fn, ks)
    for k, precision, recall in zip(ks, precisions, recalls):
        f1 = 2. * precision * recall / (precision + recall) if precision + recall else 0
        print('[k=%d: precision=%.3f%%, recall=%.3f%%, f1=%.3f%%]\n' %
              (k, 100. * precision, 100. * recall, 100. * f1), end='')
    print("*********************************************************")