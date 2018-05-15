import pandas as pd
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import pdb
import nltk
import string
from nltk.corpus import stopwords
from random import shuffle

if torch.cuda.is_available():
    print("GPU is available")
else:
    print("NO GPU !!!!")


def clean_list_of_word(words, remove_stopping_word=False):
    """remove punctuations"""
    translator = str.maketrans('', '', string.punctuation)
    result = list(map(lambda x: x.translate(translator) , words))
    
    """remove stopping word"""
    if remove_stopping_word:
        result = [word for word in result if word not in stopwords.words('english')]
    return result

words = pd.read_table("../glove.6B/glove.6B.100d.txt", sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
print("Finish loading the GloVe")
def vec(w):
  return words.loc[w].as_matrix()

def get_single_sentence_embedding(sent):
    result = []
    for item in sent:
        try:
            result.append(np.array(vec(item)))
        except:
            result.append(np.array(vec(",")))
            continue
    return np.array(result) 

def get_batch_embedding(sent_list):
    max_len = max(list(map(len, sent_list)))
    result = []
    for sent in sent_list:
        result.append(get_single_sentence_embedding(sent + ["."]*(max_len - len(sent))))
    return np.array(result) 


class RNNModel(nn.Module):
    def __init__(self, input_dimension, hidden_dimension, batch_size):
        super(RNNModel, self).__init__()
        self.input_dim = input_dimension
        self.hidden_dim = hidden_dimension
        self.batch_size = batch_size
        
        self.lstm = nn.LSTM(input_dimension, hidden_dimension, bidirectional=False)
        self.fc = nn.Linear(hidden_dimension, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, hidden, c):
        x = x.transpose(0, 1)
        len_seq = len(x)
        outputs, last = self.lstm(x, (hidden, c))
        output = self.fc(last[0])
        output = self.sigmoid(output)
        return output.squeeze()
           
    def init_hidden(self):
        h0 = Variable(torch.randn(1, self.batch_size, self.hidden_dim))
        c0 = Variable(torch.randn(1, self.batch_size, self.hidden_dim))
        if torch.cuda.is_available():
            h0 = h0.cuda()
            c0 = c0.cuda()
        return h0, c0

df = pd.read_csv('../all_ter_data_dropna.csv')
print("Finish loading the data")
one_list = list(df.loc[df['ht'] == 1.0]['Unnamed: 0'])
zero_list = list(df.loc[df['ht'] == 0.0]['Unnamed: 0'])
one_train = one_list[2000:16000]
validation_list = one_list[:2000] + zero_list[:2000]
shuffle(validation_list)

batch_size = 50
model = RNNModel(100, 100, batch_size)
if torch.cuda.is_available():
    print("GPU is available")
    model = model.cuda()
optimizer = optim.SGD(model.parameters(), lr=0.5)
criteria = nn.BCELoss()
model.train()

def get_batch(ids):
    x_batch = []
    y_batch = []
    for index in ids:
        row = df.loc[df['Unnamed: 0'] == index]
        label = int(float(row['ht']))
        detail = clean_list_of_word(str(row['juicy_details'].item()).split(' '), True)
        x_batch.append(detail)
        y_batch.append(label)
    x_batch = get_batch_embedding(x_batch)
    return x_batch, y_batch

def evaluate(model, ids, batch_size, thresholds=[0.4,0.5,0.6]):
    model.eval()
    count = 0
    count_right = [0,0,0]
    for i in range(int(4000/batch_size)):
        indexs = ids[i*batch_size:(i+1)*batch_size]
        x_batch, y_batch = get_batch(indexs)
        x = Variable(torch.from_numpy(x_batch).float())
        if torch.cuda.is_available():
            x = x.cuda()
        hidden, c_t = model.init_hidden()
        output = model(x, hidden, c_t)
        output = output.data.cpu().numpy()
        count += len(output)
        for j in range(len(thresholds)):
            predict = [1 if p >= thresholds[j] else 0 for p in output]
            count_right[j] += (np.array(predict) == np.array(y_batch)).sum()
    result = [c/count for c in count_right]
    return result

print("Start Training")
for e in range(1000):
    zero_train = np.random.choice(zero_list[2000:], 14000, replace=False).tolist()
    train_list = zero_train + one_train
    shuffle(train_list)
    for i in range(int(28000/batch_size)):
        ids = train_list[i*batch_size:(i+1)*batch_size]
        x_batch, y_batch = get_batch(ids)
        x = Variable(torch.from_numpy(x_batch).float())
        y = Variable(torch.from_numpy(np.array(y_batch)).float())
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
        optimizer.zero_grad()
        hidden, c_t = model.init_hidden()
        output = model(x, hidden, c_t)
        loss = criteria(output, y)
        loss.backward()
        optimizer.step()
        if i % 200 == 0:
            predict = [1 if p >= 0.5 else 0 for p in output.data.cpu().numpy()]
            training_acc = (np.array(predict) == np.array(y_batch)).sum()/len(predict)
            valid_accuracy = evaluate(model, validation_list, batch_size)
            print("Epoch: {}, Step: {}, Loss: {}, training Acc: {}, Validation Accuracy: {}".format(e, i, loss.data[0], training_acc, valid_accuracy))
            model.train()
            max_accuracy = max(valid_accuracy)
            if max_accuracy >= 0.65:
                print("Save a model")
                torch.save(model.state_dict(), "model_{}.pt".format(int(max_accuracy*100)))
    print("save model per epoch")
    torch.save(model.state_dict(), "model_epoch.pt")



