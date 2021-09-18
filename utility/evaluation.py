from dataclasses import dataclass
from typing import Tuple, List, Callable, Dict


@dataclass
class TopkData:
    test_user_item_set: dict  # 在测试集上每个用户可以参与推荐的物品集合
    test_user_positive_item_set: dict  # 在测试集上每个用户有行为的物品集合
    info_set: list
    info_positive_set: list
@dataclass
class TopkStatistic:
    hit: int = 0  # 命中数
    ru: int = 0  # 推荐数
    tu: int = 0  # 行为数

    from typing import List, Tuple
#todo  def topk_evaluate(topk_data: TopkData, score_fn: Callable[[Dict[str, List[int]]], List[float]],
def topk_evaluate(topk_data: TopkData, score_fn, ks) -> Tuple[List[float], List[float]]:
    # ks=[1, 2, 5, 10, 20, 50, 100]
    # ks = [10, 36, 64, 100]
    kv = {k: TopkStatistic() for k in ks}
    # todo
    print('kv', kv)
    print(topk_data.test_user_item_set)
    print(topk_data.test_user_positive_item_set)
    print(len(topk_data.test_user_item_set))
    print(len(topk_data.test_user_positive_item_set))
    for (user_id, item_set), d in zip(topk_data.test_user_item_set.items(), topk_data.info_set):
        ui = {'user_id': [user_id] * len(item_set), 'item_id': list(item_set)}
        item_score_list = list(zip(item_set, score_fn(ui, d[0], d[1], d[2], d[3])))
        sorted_item_list = [x[0] for x in sorted(item_score_list, key=lambda x: x[1], reverse=True)]
        positive_set = topk_data.test_user_positive_item_set[user_id]
        for k in ks:
            topk_set = set(sorted_item_list[:k])
            kv[k].hit += len(topk_set & positive_set)# 命中数
            kv[k].ru += len(topk_set)# 推荐数
            kv[k].tu += len(positive_set)# 行为数
            # print('k, kv[k].hit, kv[k].ru, kv[k].tu', k, kv[k].hit, kv[k].ru, kv[k].tu)
    return [kv[k].hit / kv[k].ru for k in ks], \
           [kv[k].hit / kv[k].tu for k in ks]  # precision, recall
