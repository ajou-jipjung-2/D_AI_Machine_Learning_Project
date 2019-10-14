import fasttext

# # Skipgram model
model = fasttext.train_unsupervised('./data/corpus_mecab_jamo.txt',"skipgram")
print(model.words)
model.save_model('model_ft_sk')