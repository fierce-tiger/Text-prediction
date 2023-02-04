import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import Counter
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import string
from keras.callbacks import LambdaCallback
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Activation
from keras.models import Sequential
from sklearn.metrics import accuracy_score

'''
ESE577 Final Project
@data: 2022/05/16
@author1: Yang Xie 114394356
@author2: Ziyan Yuan 114627023
'''

class Book2DataSet():
    def __init__(self, data_file_path, window_size):
        with open(data_file_path,'r',encoding='utf-8') as f:
            doc = f.read()
        words_tokenized = self.clean_doc(doc)
        self.context_target =  [([words_tokenized[i-(j+1)] for j in reversed(range(window_size))],words_tokenized[i])
                                  for i in range(window_size, len(words_tokenized)-window_size)]
        self.vocab = Counter(words_tokenized)
        self.word_to_idx = {word_tuple[0]: idx for idx, word_tuple in enumerate(self.vocab.most_common())}
        self.idx_to_word = list(self.word_to_idx.keys())
        self.vocab_size = len(self.vocab)
        self.window_size = window_size

    def __getitem__(self, idx):
        context = torch.tensor([self.word_to_idx[w] for w in self.context_target[idx][0]])
        target = torch.tensor([self.word_to_idx[self.context_target[idx][1]]])
        return context, target

    def __len__(self):
        return len(self.context_target)

    # turn a doc into clean tokens
    def clean_doc(self,doc):
        doc = doc.replace('”','').replace('“','')
        # split into tokens by white space
        tokens = doc.split()
        # remove punctuation from each token
        tokens = [' ' if w in string.punctuation else w for w in tokens]
        # remove remaining tokens that are not alphabetic
        tokens = [word for word in tokens if word.isalpha()]
        tokens = [word.lower() for word in tokens]
        return tokens

def print_k_nearest_neighbour(X, idx, k, idx_to_word):
    dists = np.dot((X - X[idx]) ** 2, np.ones(X.shape[1]))
    idxs = np.argsort(dists)[:k]
    print('The {} nearest neighbour of {} are: '.format(str(k), idx_to_word[idx]))
    for i in idxs:
        print(idx_to_word[i])
    return idxs

class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim, window_size):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)
        self.window_size = window_size

    def forward(self, inputs):
        embeds = torch.sum(self.embeddings(inputs), dim=1)
        out = self.linear(embeds)
        #softmax compute log probability
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

#STEP1, CBOW model training params definition
WINDOWS_SIZE = 3
EMBEDDING_DIM = 100
BATCH_SIZE = 500
NUM_EPOCH = 30
data_file_path = './data/The Adventures of Sherlock Holmes.txt'

#STEP2, get training dataset for CBOW
data = Book2DataSet(data_file_path,WINDOWS_SIZE)
print('Total sequences of tokens: %d' % len(data.context_target))
print('Unique Tokens: %d' % data.vocab_size) # 6673 unique tokens
print('The first 10 sequences of tokens: %s' % data.context_target[:10])

#STEP3, prepare CBOW training model
model = CBOW(data.vocab_size, EMBEDDING_DIM, WINDOWS_SIZE)
#optimizer = optim.SGD(model.parameters(), lr=0.001)
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_function = nn.NLLLoss()
losses = []
data_loader = DataLoader(data, batch_size=BATCH_SIZE)
cuda_available = torch.cuda.is_available()

#STEP4, start the training process
for epoch in range(NUM_EPOCH):
    total_loss = 0
    for context, target in tqdm(data_loader):
        if context.size()[0] != BATCH_SIZE:
            continue
        if cuda_available:
            context = context.cuda()
            target = target.squeeze(1).cuda()
            model = model.cuda()
        else:
            target = target.squeeze(1)
        model.zero_grad() # grad emptying
        log_probs = model(context)
        loss = loss_function(log_probs, target) #CEloss
        loss.backward() # calculate grad
        optimizer.step()
        total_loss += loss.item()
    losses.append(total_loss)
    print('epoch {}/{}; total_loss: {} '.format(str(epoch),str(NUM_EPOCH), str(total_loss)))

#STEP5, get the training results, this is the word2vec matrix of CBOW
embed_matrix = model.embeddings.weight.detach().cpu().numpy() #Standard Normal Distribution
# test the results, it will show some words that have a close relation to the word 'she', 'is', 'good'
print_k_nearest_neighbour(embed_matrix, data.word_to_idx['she'], 10, list(data.word_to_idx.keys()))
print_k_nearest_neighbour(embed_matrix, data.word_to_idx['is'], 10, list(data.word_to_idx.keys()))
print_k_nearest_neighbour(embed_matrix, data.word_to_idx['good'], 10, list(data.word_to_idx.keys()))

#STEP6, visualize the word2vec results
# convert matrix to dict {word1:word_vec1,word2:word_vec2...}
word_2_vec = {}
for word in data.word_to_idx.keys():
    word_2_vec[word] = embed_matrix[data.word_to_idx[word], :]
print("First 5 word2vec: ", list(word_2_vec.items())[:5])
#save the word2vec results
with open("./data/CBOW_en_wordvec.txt", 'w') as f:
    for key in data.word_to_idx.keys():
        f.write('\n')
        f.writelines('"' + str(key) + '":' + str(word_2_vec[key]))
    f.write('\n')
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(embed_matrix)
# reduce the word2vec size from EMBEDDING_DIM to 2 dimension
# {word1:(vec1，vec2),word2:(vec1，vec2)...}
word2ReduceDimensionVec = {}
for word in data.word_to_idx.keys():
    word2ReduceDimensionVec[word] = principalComponents[data.word_to_idx[word], :]
# plot 1000 words, words that have close relation will stay close to each other
plt.figure(figsize=(20, 20))
count = 0
for word, wordvec in word2ReduceDimensionVec.items():
    if count < 1000:
        plt.scatter(wordvec[0], wordvec[1])
        plt.annotate(word, (wordvec[0], wordvec[1]), fontsize=10)
        count += 1
plt.show()

#STEP7, preparing the training and testing data for LSTM
train_x = np.zeros([len(data.context_target), WINDOWS_SIZE], dtype=np.int32)
train_y = np.zeros([len(data.context_target)], dtype=np.int32)
i = 0
for context, target in tqdm(data):
    train_x[i] = context
    train_y[i] = target
    i += 1
print('train_x shape:', train_x.shape)
print('train_y shape:', train_y.shape)
#Pseudo-random Sequence, randomly select 1000 test data for testing
test_dataset_size = 1000
test_x = np.zeros([test_dataset_size, WINDOWS_SIZE], dtype=np.int32)
test_y = np.zeros([test_dataset_size], dtype=np.int32)
np.random.seed(42)
shuffled_indices = np.random.permutation(len(data))
test_indices = shuffled_indices[:test_dataset_size]
test_x = train_x[test_indices]
test_y = train_y[test_indices]

#STEP8, preparing LSTM model, param 'weights' is from the CBOW training results
model = Sequential()
model.add(Embedding(input_dim=data.vocab_size, output_dim=EMBEDDING_DIM, weights=[embed_matrix]))
model.add(LSTM(units=EMBEDDING_DIM))
model.add(Dense(units=data.vocab_size)) #unique tokens
model.add(Activation('softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
print(model.summary())

#get the best prediction
def best_prediction(preds, temperature=1.0):
  if temperature <= 0:
    return np.argmax(preds)
  preds = np.asarray(preds).astype('float64')
  preds = np.log(preds) / temperature
  exp_preds = np.exp(preds)
  preds = exp_preds / np.sum(exp_preds)
  probas = np.random.multinomial(1, preds, 1)
  return np.argmax(probas)

#put all best predictions is a list
def get_best_predictions(preds):
  pred_y = np.zeros([test_dataset_size], dtype=np.int32)
  for idx, pred in enumerate(preds):
      pred_y[idx] = best_prediction(pred)
  return pred_y

#give a sample of 3-words sequence and get the 4th prediction word.
def generate_next(text, num_generated=1):
  word_idxs = [data.word_to_idx[word] for word in text.lower().split()]
  for i in range(num_generated):
    x = np.array(word_idxs).reshape(1, 3)
    prediction = model.predict(x)
    idx = best_prediction(prediction[-1], temperature=0.7)
    word_idxs.append(idx)
  return ' '.join(data.idx_to_word[idx] for idx in word_idxs)

#STEP10, to get a sample 4th prediction and calculate the predict accuracy of the test dataset
def on_epoch_end(epoch, _):
  print('\nGenerating text after epoch: %d' % epoch)
  texts = [
      'my eyes tell', # my eyes tell me
      'you explain your',# you explain your process
      'hunting crop came', # hunting crop came down
      'do us some', # do us some harm
      'Then I made', # Then I made inquiries
      'he would do', # he would do nothing
  ]
  for text in texts:
    sample = generate_next(text)
    print('%s... -> %s' % (text, sample))
  #accuracy_calculation
  pred_y = get_best_predictions(model.predict(test_x))
  print('test accuracy_score: %d' % accuracy_score(test_y, pred_y))


#STEP9, LSTM model training
model.fit(train_x, train_y, batch_size=500, epochs=30, callbacks=[LambdaCallback(on_epoch_end=on_epoch_end)])
model.save("./data/lstm_train_dim100_epoch20")

