# -*- coding: utf-8 -*-

from gensim.models import fasttext as fText
from sklearn.manifold import TSNE
import matplotlib.pylab as plt
import pandas as pd
import numpy as np
from matplotlib import font_manager, rc

font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

model = fText.load_facebook_model('./fasttext/fasttext.bin')
print("load_model_success!!")

vocab_list = open("vocab_list.txt", 'r', encoding="utf-8").read().split()
X =[]
for vocab_item in vocab_list:
    X.append([])
    X[-1]=model.wv.get_vector(vocab_item)
X= np.array(X)
print(X[0])
print(X[1])
tsne = TSNE(n_components=2)

X_tsne = tsne.fit_transform(X[:,:])

df = pd.DataFrame(X_tsne, index=vocab_list[:], columns=['x', 'y'])
print(df.shape)

fig = plt.figure()
fig.set_size_inches(20,20,forward=True)
ax = fig.add_subplot(1, 1, 1)

ax.scatter(df['x'], df['y'])

for word, pos in df.iterrows():
    ax.annotate(word, pos, fontsize=15)
plt.show()