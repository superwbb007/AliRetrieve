# import pandas as pd
# from utils import post_process
# # corpus = pd.read_csv('../data/original_data/corpus.tsv')
# dev_query = pd.read_csv('../data/original_data/dev.query.txt')
# qrels_train = pd.read_csv('../data/original_data/qrels.train.tsv')
# train_query = pd.read_csv('../data/original_data/train.query.txt')


# with open('../data/original_data/train.query.txt') as f:
#     for line in f.readlines():
#         content = line.split('\t')
#         id = content[0]
#         text = content[1]
#         if text.find('\n') == -1:
#             test = text + '\n'
#         #大写转小写 + 繁体转简体
#         text = post_process(text)
#         line = (id + '\t' + text).rstrip('\n') + '\n'
#         train_query.append(line)

        
        


# with open('../data/original_data/train.query.txt', 'r') as f:
#     for line in f.readlines():
#         content = line.split('\t')
#         id = content[0]
#         text = content[1]
#         if text.find('\n') == -1:
#             test = text + '\n'
#         # 大写转小写 + 繁体转简体
#         text = post_process(text)
#         line = (id + '\t' + text).rstrip('\n') + '\n'
#         train_query.append(line)
# with open('../data/processed_data/train.query.txt', 'w') as f:
#     f.writelines(train_query)
# # print(corpus.head())
# print(dev_query.head())
# print(qrels_train.head())
# print(train_query.head())



string = "geekssss"
print(string.lstrip('s'))