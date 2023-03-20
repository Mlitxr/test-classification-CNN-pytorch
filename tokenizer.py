from config import *

def tokenize(sentence):
    for i in punctuation:
        sentence = sentence.replace(i,"")
    sentence = sentence.lower()
    word_list = sentence.split(" ")
    return word_list