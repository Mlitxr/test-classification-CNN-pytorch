import random

def load_data(path):
    with open(path) as f:
        raw_list = f.readlines()
        random.shuffle(raw_list)
        data_list=[]
        for line in raw_list:
            line = eval(line)
            data_list.append(line)
        return data_list