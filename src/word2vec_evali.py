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
print("load_model_success!!")

print(model.wv.most_similar(positive=['솔로','인공지능']))