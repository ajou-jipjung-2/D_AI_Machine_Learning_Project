from gensim.models import fasttext as fText
from sklearn.manifold import TSNE
import matplotlib.pylab as plt
import pandas as pd
import numpy as np
from matplotlib import font_manager, rc
import fasttext as ft

model = None
vocab_list = None
adj_list = None

def makeSentence(k1,min,max):
    sm_A_list = []

    for vocab_item in vocab_list:
        if model.similarity(vocab_item, k1)>min and model.similarity(vocab_item, k1)<max:
            sm_A_list.append([vocab_item, model.similarity(vocab_item, k1)])
    sm_A_list = sorted(sm_A_list, key=lambda acc: acc[1], reverse=True)
    print(sm_A_list)

def main():
    global model

    model = fText.load_facebook_model('model_ft_sk')
    print("load_model_success!!")
    vocab_list = open("label_list.txt", 'r', encoding="utf-8").read().split()
    adj_list = open("adj_list.txt", 'r', encoding="utf-8").read().split()

    k1 = '시계계'
    makeSentence(k1,0.3,0.5)

main()