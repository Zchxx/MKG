from MKG_User.utility.decorator import logger
from nltk.corpus import stopwords
from gensim.models import Word2Vec
@logger('评论文本的实体与概念的词嵌入预训练')
def d():
    c = []
    with open("KG-concept.txt", 'r', encoding='utf_8') as f:
        for line in f:
            line = line.strip().split('\t')
            concept = line[0]
            if concept in c:
                continue
            print(concept)
            c.append(concept)
    #         stops_word = set(stopwords.words("english"))
    #         concept_set = [w.lower().strip().split() for w in c if w not in stops_word]
    # model = Word2Vec(concept_set, size=300, window=3, min_count=1)
    # model.wv.save("concept_word2vec.wv")
@logger('评论文本的实体与概念的词嵌入预训练')
def g():
    lines_seen = set()
    with open("KG-concept.txt", 'r', encoding='utf_8') as f:
        for line in f:
            line = line.strip().split('\t')
            concept = line[0]
            if concept not in lines_seen:
                #outfile.write(line + '\n')
                print(concept)
                lines_seen.add(concept)
    stops_word = set(stopwords.words("english"))
    concept_set = [w.lower().strip().split() for w in lines_seen if w not in stops_word]
    print(concept_set)
if __name__ == '__main__':
    # d()
    g()