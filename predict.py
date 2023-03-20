from CNNmodel import *
from tokenizer import *
from matrix_transformer import *
import torch
from config import *

cnn_model=cnnmodel()
with open(test_data_path) as f:
    lines = f.readlines()
#print(lines[1]) #0, Considering this was shot in 1972 the video quality is excellent (4:3). Nice sharp (close-up) images with longer shots a little muddled as to be expected with only a few spotlights running during the show. Audio is in its original stereo and sounds OK to pretty good. The performance is stellar but a little short (64 Minutes). It's got a really cool trippy film section about half way through. Far-out. If you like the music of Traffic- get it while you can.
model_id = int(input("请输入模型参数编号："))
model_path = './models/model_trained_{}.pth'.format(model_id)
cnn = torch.load(model_path)
submit_path = 'submit.txt'

res=[]
for id in range(len(lines)-1):
    sentence = lines[id+1]
    num = ''
    for i in sentence:
        if i != ',':
            num = num+i
        else:
            break
    idx = int(num)
    sentence = sentence.replace(num, '')
    sentence = sentence.replace('\n', '')
    words = tokenize(sentence)
    words = list(filter(None, words))
    tensor = tranform2matrix(words)
    input = torch.reshape(torch.tensor(tensor, dtype=torch.float32), (1, padding_size, word2vec_size))
    output = cnn(input)
    #print(output.argmax(dim=1))
    res.append(int(output.argmax(dim=1)))

with open(submit_path,'w') as file:
    file.write("id, pred\n")
    for i in range(len(res)):
        file.write("{}, {}\n".format(i,res[i]))