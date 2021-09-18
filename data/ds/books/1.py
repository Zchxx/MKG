with open('Books2id.csv', 'r', encoding='utf-8') as f:
    with open(r'BX-Book-Ratings.csv', 'w', encoding='utf-8') as m:
        for line in f.readlines():
            values = line.strip().split(' , ')
            user_id, book_id, rating, context = str(values[0]), values[1], int(float(values[2])), str(values[3])
            print(user_id,',',book_id,',',rating, file=m)