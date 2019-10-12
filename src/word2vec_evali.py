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
# print(len(X[0]))
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