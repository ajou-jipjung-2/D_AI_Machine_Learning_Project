from gensim.models import fasttext as fText
import secrets

class singleton_fasttext:
    _instance = None
    model = None
    vocab_list = None
    adj_list = None
    @classmethod
    def _getInstance(cls):
        return cls._instance

    @classmethod
    def instance(cls, *args, **kargs):
        cls._instance = cls(*args, **kargs)
        cls.instance = cls._getInstance
        return cls._instance

    def __init__(self):
        global model, vocab_list, adj_list
        model = fText.load_facebook_model('./fasttext/fasttext.bin')
        print("load_model_success!!")
        vocab_list = open("label_list.txt", 'r', encoding="utf-8").read().split()
        adj_list = open("adj_list.txt", 'r', encoding="utf-8").read().split()

    def get_vocab_list(self):
        return vocab_list

    def get_adj_list(self):
        return adj_list

    def makeSentence(slef,k1, min, max):
        global model,vocab_list,adj_list
        sm_A_list = []
        for vocab_item in vocab_list:
            vocab = model.similarity(vocab_item, k1)
            if vocab > min and vocab < max:
                sm_A_list.append([vocab_item, vocab])
        sm_A_list = sorted(sm_A_list, key=lambda acc: acc[1], reverse=True)
        #     print(sm_A_list)

        select_label = secrets.choice(sm_A_list)
        print("select_label : ", select_label)

        adj_A_list = []
        adj_B_list = []
        for adj_item in adj_list:
            adj_A = model.similarity(adj_item, k1)
            adj_B = model.similarity(adj_item, select_label[0])
            adj_A_list.append([k1, select_label[0], adj_item, adj_A * adj_B])
        adj_A_list = sorted(adj_A_list, key=lambda acc: acc[3], reverse=True)
        #     print(adj_A_list)
        for sentence in adj_A_list:
            print(sentence[0] + ' ' + sentence[2] + ' ' + sentence[1])

    def makeSentence2(k1, k2, min, max):
        global model, vocab_list, adj_list
        sm_A_list = []
        for vocab_item in vocab_list:
            vocab1 = model.similarity(vocab_item, k1)
            vocab2 = model.similarity(vocab_item, k2)
            #         if vocab1>min and vocab1<max and vocab2>min and vocab2<max:
            sm_A_list.append([vocab_item, vocab1 * (1 - vocab2)])
        sm_A_list = sorted(sm_A_list, key=lambda acc: acc[1], reverse=True)
        print(sm_A_list)
        select_label = secrets.choice(sm_A_list)
        #     for item in sm_A_list:
        #         if item[0]=='신호등':
        #             select_label = item

        k3 = k1 + " " + k2
        adj_A_list = []
        adj_B_list = []
        for adj_item in adj_list:
            adj_A = model.similarity(adj_item, k3)
            adj_B = model.similarity(adj_item, select_label[0])
            adj_A_list.append([k3, select_label[0], adj_item, adj_A * adj_B])
        adj_A_list = sorted(adj_A_list, key=lambda acc: acc[3], reverse=True)
        #     print(adj_A_list)
        for sentence in adj_A_list:
            print(sentence[0] + ' ' + sentence[2] + ' ' + sentence[1])

