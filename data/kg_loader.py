import os
from typing import Dict, List, Tuple, Callable, Any
from MKG_User.data import data_loader, data_process
from MKG_User.utility.decorator import logger

# 记下kg文件夹的路径，确保其它py文件调用时读文件路径正确
kg_path = os.path.join(os.path.dirname(__file__), 'kg')

@logger('Start reading item entity mapping relationship，', ('kg_directory', 'item_id_type'))
def _read_item_id2entity_id_file(kg_directory: str, item_id_type: type = int) -> Tuple[Dict[Any, int], Dict[int, Any]]:
    item_to_entity = {}
    entity_to_item = {}
    with open(os.path.join(kg_path, kg_directory, 'item_id2entity_id.txt')) as f:
        for line in f.readlines():
            values = line.strip().split('\t')
            item_id = values[0] if item_id_type == str else item_id_type(values[0])
            entity_id = int(values[1])
            item_to_entity[item_id] = entity_id
            entity_to_item[entity_id] = item_id
    return item_to_entity, entity_to_item


@logger('Start reading the knowledge graph structure diagram，', ('kg_directory', 'keep_all_head',))
def _read_kg_file(kg_directory: str, entity_id_old2new: Dict[int, int], keep_all_head=True) ->\
        Tuple[List[Tuple[int, int, int]], List[Tuple[int, int]], list, int, int]:
    n_entity = len(entity_id_old2new)
    relation_id_old2new = {}
    n_relation = 1
    kg = []
    with open(os.path.join(kg_path, kg_directory, 'kg.txt')) as f:
        for line in f.readlines():
            values = line.strip().split('\t')
            head_old = int(values[0])
            relation_old = values[1]
            tail_old = int(values[2])
            if head_old not in entity_id_old2new:
                if keep_all_head:
                    entity_id_old2new[head_old] = n_entity
                    n_entity += 1
                else:
                    continue
            head = entity_id_old2new[head_old]
            if tail_old not in entity_id_old2new:
                entity_id_old2new[tail_old] = n_entity
                n_entity += 1
            tail = entity_id_old2new[tail_old]
            if relation_old not in relation_id_old2new:
                relation_id_old2new[relation_old] = n_relation
                n_relation += 1
            relation = relation_id_old2new[relation_old]
            kg.append((head, relation, tail))
    # todo 权重异构信息图
    r_relation = []
    relation_name_list = []
    with open(os.path.join(kg_path, kg_directory, 'kg.txt')) as f:
        for line in f.readlines():
            values = line.strip().split('\t')
            relation_name = values[1]
            relation_name_list.append(relation_name)
            if relation_name not in r_relation:
                r_relation.append(relation_name)
        n = 1
        print(r_relation)
        for i in range(len(r_relation)):
            r_relation[i] = n
            n += 1
            i += 1
    print(r_relation)
    print(len(r_relation))
    print(len(relation_name_list))
    print(type(relation_old))
    relation_weight = {}
    for key in relation_name_list:
        relation_weight[key] = relation_weight.get(key, 1) + 1
    relation_weight = list(relation_weight.values())
    relation_weight = [relation_weight[i] / len(relation_name_list) for i in range(len(r_relation))]
    print(relation_weight)
    dict_relation2weight =dict(zip(r_relation, relation_weight))
    print(dict_relation2weight)
    #得到的是知识图谱中关系种类和对应的数量的列表类型
    relation2weight = []
    for i in range(len(r_relation)):
        relation2weight.append((r_relation[i], relation_weight[i]))
        i += 1
    print(relation2weight)

    return kg, relation2weight, relation_weight, n_entity, n_relation


@logger('----------Start loading the data set with knowledge graph：', end_message='----------The data set with the knowledge graph is loaded', log_time=False)
def _read_data_with_kg(kg_loader_config: Tuple[str, Callable[[], List[tuple]], type],
                       negative_sample_ratio=1, negative_sample_threshold=0, negative_sample_method='random',
                       keep_all_head=True) -> Tuple[List[Tuple[int, int, int, float, float, float]], List[Tuple[int, int, int]],
                                                    List[Tuple[int, int]], list, int, int, int, int]:
    kg_directory, data_loader_fn, item_id_type = kg_loader_config
    # 开始读物品实体映射关系
    old_item_to_old_entity, old_entity_to_old_item = _read_item_id2entity_id_file(kg_directory, item_id_type)
    # 读数据
    data = data_loader_fn()# 原始数据
    # 将不存在的向量进行映射
    old_entity_to_old_item = {old_entity: d[1] for old_entity, old_item in old_entity_to_old_item.items()
                              for d in data if d[1] == old_item}
    # data的数据都在item_id2entity_id
    data = [d for d in data if d[1] in old_item_to_old_entity]  # 去掉知识图谱中不存在的物品
    # 采集负样本
    data = data_process.negative_sample(data, negative_sample_ratio, negative_sample_threshold, negative_sample_method)
    #id规整化
    data, n_user, n_item, _, item_id_old2new = data_process.neaten_id(data)
    # todo 词向量预处理
    data = data_process.word2embedding(data)
    
    entity_id_old2new = {old_entity: item_id_old2new[old_item]
                         for old_entity, old_item in old_entity_to_old_item.items()}
    # 读取知识图谱结构图
    kg, relation2weight, relation_weight, n_entity, n_relation = _read_kg_file(kg_directory, entity_id_old2new, keep_all_head)
    return data, kg, relation2weight, relation_weight, n_user, n_item, n_entity, n_relation


# kg_loader_configs: (kg_directory, data_loader_fn, item_id_type)
ml1m_kg3m = 'ml3m-kg', data_loader.ml3m, int
books_kg = 'books-kg', data_loader.books, str
music_kg = 'music-kg', data_loader.music, int

if __name__ == '__main__':
    data, kg,  relation2weight, n_user, n_item, n_entity, n_relation = _read_data_with_kg(books_kg)
