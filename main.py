"""
Name：多类型知识图谱的用户画像推荐系统
Author :邹程辉
Time: 2020.7
"""
from MKG_User.tool import get_user_positive_item_list, construct_directed_kg, get_relation_weight, get_kg_set
from MKG_User.model import KG_User_model
from MKG_User.train import train
from MKG_User.data import kg_loader, data_process
import tensorflow as tf

if __name__ == '__main__':

    n_user, n_item, n_entity, n_relation, train_data, test_data, eval_data, \
    kg, relation2weight, relation_weight, topk_data\
        = data_process.pack_kg(kg_loader.music_kg, negative_sample_threshold=4, split_ensure_positive=True)
    hop_size, kg_size = 3, 32

    # 基于知识图谱得到用户历史项目在知识图谱中的扩展
    kg_set = get_kg_set(n_user, hop_size, kg_size, get_user_positive_item_list(train_data),
                        construct_directed_kg(kg), get_relation_weight(relation2weight), relation_weight)
    # 模型
    model = KG_User_model(n_entity, n_relation, kg_set, hop_size, kg_size, dim=16, kge_weight=0.01, l2=1e-7)
    #训练
    train(model, train_data, test_data, eval_data, topk_data, optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
          epochs=30, batch=512)

