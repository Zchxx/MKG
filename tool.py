from MKG_User.utility.decorator import logger
from typing import List, Tuple, Dict
from collections import defaultdict
import numpy as np
import math

@logger('得到每个用户在训练集上的正反馈物品集合')
def get_user_positive_item_list(train_data: List[Tuple[int, int, int, float, float, float]]) -> Dict[int, List[int]]:
    user_positive_item_list = defaultdict(list)
    #todo
    for user_id, item_id, label, context, review_entity, concept in train_data:
        if label == 1:
            user_positive_item_list[user_id].append(item_id)
    return user_positive_item_list


@logger('根据知识图谱结构构建有向图')
def construct_directed_kg(kg: List[Tuple[int, int, int]]) -> Dict[int, List[Tuple[int, int]]]:
    kg_dict = defaultdict(list)
    for head_id, relation_id, tail_id in kg:
        kg_dict[head_id].append((relation_id, tail_id))
    return kg_dict

@logger('得到知识图谱的关系权重')
def get_relation_weight(relation2weight: List[Tuple[int, int]]) -> Dict[int, List[Tuple[int, int]]]:
    relation2weight_dict = defaultdict(list)
    for relation_name, relation_weight in relation2weight:
        relation2weight_dict[relation_name].append((relation_weight))
    return relation2weight_dict

@logger('根据知识图谱有向权重图得到每个用户每跳的三元组，', ('n_user', 'hop_size', 'kg_size'))
def get_kg_set(n_user: int, hop_size: int, kg_size: int, user_positive_item_list: Dict[int, List[int]],
               kg_dict: Dict[int, List[Tuple[int, int]]], relation2weight_dict: Dict[int, List[Tuple[int]]],
               relation_weight: list) -> List[List[Tuple[List[int], List[int], List[int]]]]:
    kg_set = [[] for _ in range(n_user)]
    for user_id, positive_item_list in user_positive_item_list.items():
        for hop in range(hop_size):
            kg_h, kg_r, kg_t = [], [], []
            tails_of_last_hop = positive_item_list if hop == 0 else kg_set[user_id][-1][2]
            for entity_id in tails_of_last_hop:
                for relation_id, tail_id in kg_dict[entity_id]:
                    kg_h.append(entity_id)
                    for relation_id in relation2weight_dict.keys():
                        relation2weight_dict_keys = list(relation2weight_dict.keys())
                        w_index = relation2weight_dict_keys.index(relation_id)
                        r_weight = math.ceil(10 * relation_weight[w_index])
                        kg_r.append(relation_id * r_weight)
                    kg_t.append(tail_id)
            # 控制了大小
            if len(kg_h) == 0:  # 如果当前用户当前跳的实体关系集合是空的
                kg_set[user_id].append(kg_set[user_id][-1])  # 仅复制上一跳的集合
            else:
                replace = len(kg_h) < kg_size
                indices = np.random.choice(len(kg_h), size=kg_size, replace=replace)
                kg_h = [kg_h[i] for i in indices]
                kg_r = [kg_r[i] for i in indices]
                kg_t = [kg_t[i] for i in indices]
                kg_set[user_id].append((kg_h, kg_r, kg_t))
    return kg_set
