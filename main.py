from CNNmodel import cnnmodel
from config import *
from dataloader import *
from tokenizer import *
from gensim.models.word2vec import Word2Vec
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter


# load_data
data_set = load_data(train_data_path)
#print(data_set[0]["label"]) #8


#tokenize the texts
sentences=[]
for data in data_set:
    sentence = data["raw"]
    sentence = tokenize(sentence)
    data["text"] = sentence
    sentences.append(sentence)
    temp_label=[0]*10
    temp_label[data["label"]] = 1
    data["label"] = temp_label
#print(data_set[0]["label"])


#word to vector
word2vec_model=Word2Vec(sentences, vector_size=word2vec_size, min_count=1)
word2vec_model.save('./models/w2v.m')
word2vec_model = Word2Vec.load('./models/w2v.m')
#print(word2vec_model.wv['happy'])


#sentences transform to matrix
from matrix_transformer import *
for data in data_set:
    data["matrix"]=tranform2matrix(data["text"])
#print(data_set[0]["matrix"])


#divide train_data and test_data
test_data=[]
train_data=[]
for i in range(len(data_set)):
    if i%divident == 0:
        test_data.append(data_set[i])
    else:
        train_data.append(data_set[i])



cnn = cnnmodel()#[1,128,128]
loss_function = nn.CrossEntropyLoss()
learning_rate = lr
opti = torch.optim.SGD(cnn.parameters(),lr=learning_rate)
train_step = 0
test_step = 0



writer = SummaryWriter("log_train")
input = data_set[0]["matrix"]
input = torch.reshape(torch.tensor(input, dtype=torch.float32), (1, padding_size, word2vec_size))
output = cnn(input)
target = torch.tensor(data_set[0]["label"])
target = target.view(1,-1)


for i in range(epoch):
    print("-----epoch: {}-----".format(i))
    cnn.train()
    for data in train_data:
        input = data["matrix"]
        input = torch.reshape(torch.tensor(input, dtype=torch.float32), (1, padding_size, word2vec_size))
        target = torch.tensor(data["label"], dtype=torch.float32)
        target = target.view(1, -1)

        opti.zero_grad()
        output = cnn(input)

        loss = loss_function(output, target)
        loss.backward()
        opti.step()

        train_step = train_step + 1
        if train_step % 100 == 0:
            print("train step: {}, loss: {}".format(train_step, loss.item()))
            writer.add_scalar("train loss", loss.item(), train_step)

    cnn.eval()
    test_loss = 0
    accuracy = 0
    correct = 0
    count = 0
    for data in test_data:
        input = data["matrix"]
        input = torch.reshape(torch.tensor(input, dtype=torch.float32), (1, padding_size, word2vec_size))
        output = cnn(input)
        target = torch.tensor(data["label"], dtype=torch.float32)
        target = target.view(1, -1)

        loss = loss_function(output, target)
        test_loss = test_loss + loss.item()
        if output.argmax(dim=1)==target.argmax(dim=1):
            correct=correct+1
        else:
            count = count+1
        #if output.argmax(dim=1).eq(target).sum().float().item() <= 1:
        #    print(output.argmax(dim=1))
        #correct = correct + output.argmax(dim=1).float().item()
    print("test accuracy: {}".format(correct/len(test_data)))
    print("test loss: {}".format(test_loss))
    print("false count: {}".format(count/len(test_data)))
    writer.add_scalar("test_loss", test_loss, test_step)
    writer.add_scalar("test_accuracy_rate", correct/len(test_data), test_step)
    test_step = test_step + 1
    torch.save(cnn, "./models/model_trained_{}.pth".format(i))
