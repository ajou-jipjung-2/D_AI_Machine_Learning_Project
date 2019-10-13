from gensim.models import fasttext as fText
from sklearn.manifold import TSNE
import matplotlib.pylab as plt
import pandas as pd
import numpy as np
from matplotlib import font_manager, rc
import fasttext as ft

font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

model = ft.load_model('./fasttext/fasttext.bin');
model2 = fText.load_facebook_model('./fasttext/fasttext.bin')
print("load_model_success!!")

# print(model.get_nearest_neighbors('핸드폰'))
# print(model2.(positive=['소리','물건']))
# model2.wv.similar_by_word('소리',topn=100)
# model2.wv.most_similar(positive=['소리','물건'])

vocab_list = open("label_list.txt", 'r', encoding="utf-8").read().split()
for vocab_item in vocab_list:
    print(vocab_item)
# f.close()
print(len(vocab_list))

sm_A_list=[]
sm_B_list =[]
sm_AB_list =[]
k1 = '컴퓨터'
k2 = '카메라'
for vocab_item in vocab_list:
    sm_A_list.append([vocab_item,model2.similarity(vocab_item,k1)])
    sm_B_list.append([vocab_item, model2.similarity(vocab_item, k2)])
    sm_AB_list.append([vocab_item,(sm_A_list[-1][1]*0.7+sm_B_list[-1][1]*0.3)])

sm_AB_list = sorted(sm_AB_list,key=lambda acc:acc[1],reverse=True)
print(sm_AB_list)