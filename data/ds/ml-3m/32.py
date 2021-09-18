import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000
with open("movies2id.csv", 'r', encoding='utf-8') as r:
    num = []
    r_len = []
    n = 0
    for line in r.readlines():
        values = line.strip().split(' , ')
        review = [w for w in values[3].strip().split()]
        print(review)
        print(len(review))
        num.append(n)
        r_len.append(len(review))
        n += 1
    plt.plot(num, r_len)
    plt.title('Movies')
    plt.savefig('movies.png', format='png', dpi=1000)
    plt.show()