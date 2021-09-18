import os
from typing import List, Callable, Tuple
from MKG_User.utility.decorator import logger

# 记下ds文件夹的路径，确保其它py文件调用时读文件路径正确
ds_path = os.path.join(os.path.dirname(__file__), 'ds')

def _read_m3(relative_path: str,   separator: str) -> List[Tuple[str, int, int, str]]:
    data = []
    with open(os.path.join(ds_path, relative_path), 'r') as f:
        for line in f.readlines():
            values = line.strip().split(separator)
            user_id, movie_id, rating, context = str(values[0]), int(values[1]), int(float(values[2])), str(values[3])
            #32121 22525 38564 80598 212854
            #56174 107557 261197 348971 226310
            #376887 376887
            #93571--93210
            data.append((user_id, movie_id, rating, context))
    return data

def _read_ml3m() -> List[Tuple[str, int, int, str]]:
    return _read_m3('ml-3m/movies2id.csv', ' , ')


def _read_music() -> List[Tuple[str, int, int, str]]:
    data = []
    with open(os.path.join(ds_path, 'music/Music2id.csv'), 'r') as f:
        for line in f.readlines():
            values = line.strip().split(' , ')
            user_id, artist_id, weight, context = str(values[0]), int(values[1]), int(float(values[2])), str(values[3])
            data.append((user_id, artist_id, weight, context))
    return data

def _read_books() -> List[Tuple[str, str, int, str]]:
    data = []
    with open(os.path.join(ds_path, 'books/Books2id.csv'), 'r', encoding='utf-8') as f:
        for line in f.readlines():
            values = line.strip().split(' , ')
            user_id, book_id, rating, context = str(values[0]), values[1], int(float(values[2])), str(values[3])
            data.append((user_id, book_id, rating, context))
    return data


@logger('开始读数据，', ('data_name', 'expect_length', 'expect_user', 'expect_item'))
def _load_data(read_data_fn: Callable[[], List[tuple]], expect_length: int, expect_user: int, expect_item: int,
               data_name: str) -> List[tuple]:
    data = read_data_fn()
    n_user, n_item = len(set(d[0] for d in data)), len(set(d[1] for d in data))
    #断言
    assert len(data) == expect_length, data_name + ' length ' + str(len(data)) + ' != ' + str(expect_length)
    assert n_user == expect_user, data_name + ' user ' + str(n_user) + ' != ' + str(expect_user)
    assert n_item == expect_item, data_name + ' item ' + str(n_item) + ' != ' + str(expect_item)
    return data

def ml3m() -> List[Tuple[int, int, int, int, str]]:
    return _load_data(_read_ml3m, 386475, 207333, 2586, 'ml3m')

def books()-> List[Tuple[str, str, int, str]]:
    return _load_data(_read_books, 1512287, 764936, 51333, 'Books')
    # return _load_data(_read_books, 1149780, 105283, 340555, 'Books')

def music()-> List[Tuple[str, int, int, str]]:
    return _load_data(_read_music, 323594, 116937, 3547, 'Music')
    # return _load_data(_read_music, 92834, 1892, 17632, 'Music')

# 测试数据读的是否正确
if __name__ == '__main__':
    pass
    #print(data)