import os
import random
import numpy as np
from typing import Tuple, List, Callable
from collections import defaultdict
from MKG_User.utility.evaluation import TopkData
from MKG_User.utility.decorator import logger
from nltk.corpus import stopwords  # 停用词
import re
from gensim.models import Word2Vec, KeyedVectors

@logger('Concept word embedding pre-training of text entities')
def concept_entity_embedding():
    concept_set = set()
    with open("KG-concept.txt", 'r', encoding='utf_8') as f:
        for line in f:
            line = line.strip().split('\t')
            concept = line[0]
            if concept not in concept_set:
                concept_set.add(concept)
    stops_word = set(stopwords.words("english"))
    concept_sen = [w.lower().strip().split() for w in concept_set if w not in stops_word]
    model = Word2Vec(concept_sen, size=300, window=3, min_count=1)
    model.wv.save("concept_word2vec.wv")

def concept_entity(sentences, w2v_model, concept_entity_list, con_w2v_model, entity_concept):
    def build_concept_entity_vector(sentence, size, w2v_model):
        sen_vec = np.zeros(size).reshape((1, size))
        count = 0
        for word in sentence:
            try:
                sen_vec += w2v_model[word].reshape((1, size))
                count += 1
            except KeyError:
                continue
        if count != 0:
            sen_vec /= count
        return sen_vec
    # Todo print(sentences)
    # entity_list = []
    # for e in sentences:
    #     if e in concept_entity_list:
    #         entity_list.append(e)
    #         if len(entity_list) == 50:
    #             continue
    entity_list = [e for e in sentences if e in concept_entity_list]
    if len(entity_list) == 0:
        entity_list.append('film')
    concept_list = [entity_concept[e] for e in entity_list]
    entity_set = [e for entity in entity_list for e in entity.strip().split()]
    concept_set =[c for concept in concept_list for c in concept.lower().strip().split()]
    concept_vec = build_concept_entity_vector(concept_set, 300, con_w2v_model)
    review_entity_vec = build_concept_entity_vector(entity_set, 300, w2v_model)
    # concept_vec = con_w2v_model[[c for concept in concept_list for c in concept.lower().strip().split()]]
    # review_entity_vec = w2v_model[[e for entity in entity_list for e in entity.strip().split()]]
    return review_entity_vec, concept_vec

@logger('Pre-training of word vector embedding of comment text and concept')
def word2embedding(data: List[tuple]):
    def clean_text(review):
        delete_word = set()
        for word in review:
            if word not in delete_word:
                delete_word.add(word)
        raw_text = " ".join(delete_word)
        text = re.sub("[^a-zA-Z]", " ", raw_text)
        text_lower = text.lower().strip().split()

        stops_word = set(stopwords.words("english"))
        delete_stops_word = [w for w in text_lower if w not in stops_word]
        return delete_stops_word
        # return " ".join(delete_stops_word)
    # 词向量的训练
    # sentences = [clean_text(d[3]) for d in data]
    # model = Word2Vec(sentences, size=300, window=3, min_count=1)
    # model.wv.save("music_word2vec.wv")#不同数据这里用的不同的向量
    # print('successful')
    # concept_entity_embedding() # 训练概念的词向量
    # 对每个句子的所有词向量取均值，来生成一个句子的vector
    def build_sentence_vector(sentence, size, w2v_model):
        sen_vec = np.zeros(size).reshape((1, size))
        count = 0
        for word in sentence:
            try:
                sen_vec += w2v_model[word].reshape((1, size))
                count += 1
            except KeyError:
                continue
        if count != 0:
            sen_vec /= count
        return sen_vec
    w2v_model = KeyedVectors.load("music_word2vec.wv", mmap='r')
    # TODO 词典的按照数据集不同进行选择
    con_w2v_model = KeyedVectors.load("concept_word2vec.wv", mmap='r')
    concept_entity_set = {}
    with open("KG-concept.txt", 'r', encoding='utf_8') as f:
        for line in f:
            line = line.strip().split('\t')
            concept = line[0]
            entity = line[1]
            concept_entity_set[concept] = entity
    concept_entity_list = set()
    for c in concept_entity_set.values():
        if c not in concept_entity_list:
            concept_entity_list.add(c)
    entity_concept = {v: k for k, v in concept_entity_set.items()}
    new_data = []
    n = 0
    #todo 
    # m = open(r'music_data.txt', 'w', encoding='utf8')
    for d in data:
        sentence = clean_text(d[3])
        sen_vec = build_sentence_vector(sentence, 300, w2v_model)
        review_entity_vec, concept_vec = concept_entity(sentence, w2v_model,
                                                        concept_entity_list, con_w2v_model, entity_concept)
        n += 1
        new_data.append((d[0], d[1], d[2], sen_vec, review_entity_vec, concept_vec))
        # print(d[0], d[1], d[2], sen_vec, review_entity_vec, concept_vec, file=m)
    print("successful", n)
    return new_data

@logger('Start collecting negative samples,', ('ratio', 'threshold', 'method'))
def negative_sample(data: List[tuple], ratio=1, threshold=4, method='random') -> List[tuple]:

    # 对负样本采集权重，均匀随机采样
    if method == 'random':
        negative_sample_weight = {d[1]: 1 for d in data}#均匀随机采样，将所有的权重设置为1
    elif method == 'popular':
        negative_sample_weight = {d[1]: 0 for d in data}
        for d in data:
            negative_sample_weight[d[1]] += 1
    else:
        raise ValueError("参数method必须是'random'或'popular'")

    # 得到每个用户正样本与非正样本集合
    user_positive_set, user_unpositive_set = defaultdict(set), defaultdict(set)
    user_positive_review_set, user_unpositive_review_set = defaultdict(set), defaultdict(set)
    for d in data:
        #118333
        #187142
        user_id, item_id, weight, context = d[0], d[1], d[2], d[3]
        (user_positive_set if weight >= threshold else user_unpositive_set)[user_id].add((item_id))
        (user_positive_review_set if weight >= threshold else user_unpositive_review_set)[user_id].add((context))
    # 仅为有正样例的用户采集负样例
    #TODO list
    user_list = set(user_positive_set.keys())
    arg_positive_set = [user_positive_set[user_id] for user_id in user_list]
    arg_unpositive_set = [user_unpositive_set[user_id] for user_id in user_list]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=os.cpu_count()//2, initializer=_negative_sample_init, initargs=(ratio, negative_sample_weight)) as executor:
        sampled_negative_items = executor.map(_negative_sample, arg_positive_set, arg_unpositive_set, chunksize=100)

    # 构建新的数据集
    new_data = []
    for user_id, negative_items, context in zip(user_list, sampled_negative_items, user_unpositive_review_set.values()):
        new_data.extend([(user_id, item_id, 0, context) for item_id in negative_items])
    for (user_id, positive_items), context in zip(user_positive_set.items(), user_positive_review_set.values()):
        new_data.extend([(user_id, item_id, 1, context) for item_id in positive_items])

    return new_data#得到的新数据将按用户对项目的的评分分为了0和1的CTR点击预测
def _negative_sample_init(_ratio, _negative_sample_weight):
    global item_set, ratio, negative_sample_weight
    item_set, ratio, negative_sample_weight = set(_negative_sample_weight.keys()),  _ratio, _negative_sample_weight
def _negative_sample(positive_set, unpositive_set):
    valid_negative_list = list(item_set - positive_set - unpositive_set)
    n_negative_sample = min(int(len(positive_set) * ratio), len(valid_negative_list))
    if n_negative_sample <= 0:
        return []
    weights = np.array([negative_sample_weight[item_id] for item_id in valid_negative_list], dtype=np.float)
    weights /= weights.sum()
    sample_indices = np.random.choice(range(len(valid_negative_list)), n_negative_sample, False, weights)
    return [valid_negative_list[i] for i in sample_indices]


@logger('Start id regularization')
def neaten_id(data: List[tuple]) -> Tuple[List[Tuple[int, int, int, str]], int, int, dict, dict]:
    new_data = []
    n_user, n_item = 0, 0
    user_id_old2new, item_id_old2new = {}, {}
    for user_id_old, item_id_old, label, context in data:
        if user_id_old not in user_id_old2new:
            user_id_old2new[user_id_old] = n_user
            n_user += 1
        if item_id_old not in item_id_old2new:
            item_id_old2new[item_id_old] = n_item#dict
            n_item += 1
        new_data.append((user_id_old2new[user_id_old], item_id_old2new[item_id_old], label, context))
    return new_data, n_user, n_item, user_id_old2new, item_id_old2new


@logger('Start data segmentation，', ('test_ratio', 'shuffle', 'ensure_positive'))
def split(data: List[tuple], test_ratio=0.4, shuffle=True, ensure_positive=True) -> Tuple[List[tuple], List[tuple]]:
    """
       将数据切分为训练集数据和测试集数据

       :param data: 原数据，第一列为用户id，第二列为物品id，第三列为标签
       :param test_ratio: 测试集数据占比，这个值在0和1之间
       :param shuffle: 是否对原数据随机排序
       :param ensure_positive: 是否确保训练集每个用户都有正样例
       :return: 训练集数据和测试集数据
       """
    if shuffle:
        random.shuffle(data)
    n_test = int(len(data) * test_ratio)
    test_eval_data, train_data = data[:n_test], data[n_test:]
    a = 0
    b = 0
    for d in test_eval_data:
        # todo
        if d[2] == 0:
            a += 1
        else:
            b += 1
    print(a, b)
    if ensure_positive:
        #user_set代表了数据集中用户减去训练数据集中正样本
        user_set = {d[0] for d in data} - {user_id for user_id, _, label, context, review_entity, concept in train_data if label == 1}#todo
        if len(user_set) > 0:
            print('警告：为了确保训练集数据每个用户都有正样例，%d(%f%%)条数据从测试集随机插入训练集'
                  % (len(user_set), 100 * len(user_set) / len(data)))

        i = len(test_eval_data) - 1
        while len(user_set) > 0:
            assert i >= 0, '无法确保测试集每个用户都有正样例，因为存在没有正样例的用户：' + str(user_set)
            if test_eval_data[i][0] in user_set and test_eval_data[i][2] == 1:
                user_set.remove(test_eval_data[i][0])
                train_data.insert(random.randint(0, len(train_data)), test_eval_data.pop(i))
            i -= 1
    return train_data, test_eval_data
@logger('Start data segmentation，', ('eval_ratio', 'shuffle', 'ensure_positive'))
def split_eval(data: List[tuple], eval_ratio=0.5, shuffle=True, ensure_positive=True) -> Tuple[List[tuple], List[tuple]]:
    if shuffle:
        random.shuffle(data)
    n_eval = int(len(data) * eval_ratio)
    test_data, eval_data = data[:n_eval], data[n_eval:]
    return test_data, eval_data

@logger('Start preparing topk evaluation data', ('n_sample_user',))
def prepare_topk(train_data: List[Tuple[int, int, int, float, float, float]],
                 test_eval_data: List[Tuple[int, int, int, float, float, float]],
                 n_user: int, n_item: int, n_sample_user=None) -> TopkData:
    if n_sample_user is None or n_sample_user > n_user:
        n_sample_user = n_user

    user_set = np.random.choice(range(n_user), n_sample_user, False)
    def get_user_item_set(data: List[Tuple[int, int, int, float, float, float]], only_positive=False):
        user_item_set = {user_id: set() for user_id in user_set}
        info_set = []
        for user_id, item_id, label, context, re_vec, con_vec in data:
            if user_id in user_set and (not only_positive or label == 1):
                user_item_set[user_id].add(item_id)
                info_set.append((label, context, re_vec, con_vec))#jia
        return user_item_set, info_set

    user_item_set, info_set = get_user_item_set(train_data)
    test_user_item_set = {user_id: set(range(n_item))
                          for user_id, item_set in user_item_set.items()}# - item_set
    test_user_positive_item_set, info_positive_set = get_user_item_set(test_eval_data)# , only_positive=True
    print(test_user_positive_item_set)
    print(user_item_set)
    return TopkData(test_user_item_set, test_user_positive_item_set, info_set, info_positive_set)


def pack_kg(kg_loader_config: Tuple[str, Callable[[], List[tuple]], type], keep_all_head=True,
            negative_sample_ratio=1, negative_sample_threshold=0, negative_sample_method='random',
            split_test_ratio=0.4, shuffle_before_split=True, split_ensure_positive=True,
            topk_sample_user=100) -> Tuple[int, int, int, int, List[Tuple[int, int, int, float, float, float]],
                                           List[Tuple[int, int, int, float, float, float]],
                                           List[Tuple[int, int, int, float, float, float]],
                                           List[Tuple[int, int, int]],
                                           List[Tuple[int, int]], list, TopkData]:
    # Tuple[int, int, int, int, List[Tuple[int, int, int, float, float, float]],
    # 调用库
    from MKG_User.data.kg_loader import _read_data_with_kg
    data, kg, relation2weight, relation_weight, n_user, n_item, n_entity, n_relation = _read_data_with_kg(
        kg_loader_config, negative_sample_ratio, negative_sample_threshold, negative_sample_method, keep_all_head)

    # 切分数据
    train_data, test_eval_data = split(data, split_test_ratio, shuffle_before_split, split_ensure_positive)
    test_data, eval_data = split_eval(test_eval_data, 0.5, shuffle_before_split, split_ensure_positive)
    # 预处理topk数据
    # topk_data = prepare_topk(train_data, test_data, eval_data, n_user, n_item, topk_sample_user)
    topk_data = prepare_topk(train_data, train_data, n_user, n_item, topk_sample_user)
    # 返回得到的数据
    return n_user, n_item, n_entity, n_relation, train_data, test_data, eval_data, kg, relation2weight, relation_weight, topk_data
