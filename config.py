import argparse


parser = argparse.ArgumentParser(description="CNN model training")
parser.add_argument('-lr', type=float, default=0.001, help='learning_rate')
parser.add_argument('-epoch', type=int, default=50, help='training eochs')
parser.add_argument('-kernel_nums', type=int, default=128,help='number of each kind of kernel')
parser.add_argument('-divident', type=int, default=5, help='ratio of the training set to the test set')
parser.add_argument('-padding_size', type=int, default=128, help='number of words picked in each sentence')
parser.add_argument('-test_data_path', type=str, default='dataset/test.txt', help='address of data used to train')
parser.add_argument('-word2vec_size',type=int, default=128, help='word2vec dim')
parser.add_argument('-dropout', type=float, default=0.5, help='dropout')
args = parser.parse_args()


train_data_path = 'dataset/train_data.txt'
test_data_path = 'dataset/test.txt'
submit_path = 'submit.txt'

punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'

padding_size = args.padding_size #(每一句的前多少个单词)
word2vec_size = args.word2vec_size #词向量的大小

divident = args.divident #训练集和测试集分割

kernel_num = args.kernel_nums
kernel_sizes = [3, 4, 5]
dropout = args.dropout
lr = args.lr

epoch = args.epoch