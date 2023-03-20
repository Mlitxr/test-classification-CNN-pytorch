from config import *
from gensim.models.word2vec import Word2Vec

word2vec_model = Word2Vec.load('./models/w2v.m')
def tranform2matrix(sentence):
    matrix=[]
    for i in range(padding_size):
        try:
            matrix.append(word2vec_model.wv[sentence[i]].tolist())
        except:
            matrix.append([0]*word2vec_size)
    return matrix