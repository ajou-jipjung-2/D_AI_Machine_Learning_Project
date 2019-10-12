import fasttext
# model = fasttext.load_model('./fasttext/fasttext.bin')
# print(model.words)
# print(model.labels)
raw_corpus_fname = '' # Fill your corpus file
model_fname = './fasttext/fasttext.bin'      # Fill your model file

skipgram_model = fasttext.cbow(
    raw_corpus_fname,
    model_fname,
    loss = 'hs',        # hinge loss
    ws=1,               # window size
    lr = 0.01,          # learning rate
    dim = 150,          # embedding dimension
    epoch = 5,          # num of epochs
    min_count = 10,     # minimum count of subwords
    encoding = 'utf-8', # input file encoding
    thread = 6          # num of threads
)