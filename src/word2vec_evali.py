
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pylab as plt
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

model = Word2Vec.load("./word2vec/word2vec")
# print(model.most_similar(positive=['커피','위해']))
# print(model.similar_by_vector('커피'))
vocab = list(model.wv.vocab)
X = model[vocab]
print(model.wv.get_vector('콜라').shape)
# print(len(X))
print(len(X[0]))
# print(X[0][:10])
# tsne = TSNE(n_components=2)
#
# X_tsne = tsne.fit_transform(X[:100,:])
#
# df = pd.DataFrame(X_tsne, index=vocab[:100], columns=['x', 'y'])
# print(df.shape)
#
# fig = plt.figure()
# fig.set_size_inches(20,20,forward=True)
# ax = fig.add_subplot(1, 1, 1)
#
# ax.scatter(df['x'], df['y'])
#
# for word, pos in df.iterrows():
#     ax.annotate(word, pos, fontsize=15)
# plt.show()

# print(model.wv.words)

vocab_list200 = ['1', '.', '0', '2', '의', ',', '이', '는', '다', ')', '(', '9', '년', '에', '을', '하', '3', '5', '4', '8', '은', '6', '7', ':', '를', '월', '분류', '일', '고', '-', '가', '있', '에서', '으로', '로', '한', '되', '었', '과', '들', '와', '도', '했', '적', '인', '였', '그', '어', '기', '《', '제', '것', '*', '~', '게', '지', '"', '》', '여', '한다', '수', '역', '된', '등', '/', '며', '대', '·', '회', '선수', '영화', '대한민국', '할', '던', '해', '아', '만', '%', '명', '않', '자', '시', '에게', '중', '주', '까지', '미국', '았', '나', '번', '면', '지만', '일본', '없', '사람', '받', '성', '위', '때', '축구', '전', '으며', '#', '된다', '세', '개', '부터', '후', '화', '라고', '사용', '호', '같', '말', '라는', '팀', '학교', '이후', '및', '?', '는데', '두', '면서', '선', '한국', '차', '라', '대학교', '서울', '–', '보', '대한', '권', '상', '경기', '|', '지역', '군', '리그', '다고', '!', '때문', '국가', '이름', '시작', '더', '함께', '세계', '다른', '내', '경우', '현재', '살', '위해', '많', '방송', 'KBS', 'of', '부', '여자', '자신', '기록', '조선', '또한', '대표', 'A', '안', '다음', '째', '대학', '사', '당시', '게임', '분', '형', '오', '하나', '신', '서', '다는', '배우', '이나', '가지', 'm', '감독', '고등학교', '출신', '작품', '가장', '씨', '계', '구', '시즌', '드라마', '활동', '남', '당', '그러나', '+']
X =[]
for vocab_item in vocab_list200:
    X.append([])
    X[-1]=model.wv.get_vector(vocab_item)
# print(X[0][:10])
X= np.array(X)
print(X[0])
print(X[1])
# # print(model.wv.get_vector(vocab_list[0]))
# # print(model.wv.get_vector(vocab_list[0]).shape)
tsne = TSNE(n_components=2)

X_tsne = tsne.fit_transform(X[:,:])

df = pd.DataFrame(X_tsne, index=vocab_list200[:], columns=['x', 'y'])
print(df.shape)

fig = plt.figure()
fig.set_size_inches(20,20,forward=True)
ax = fig.add_subplot(1, 1, 1)

ax.scatter(df['x'], df['y'])

for word, pos in df.iterrows():
    ax.annotate(word, pos, fontsize=15)
plt.show()