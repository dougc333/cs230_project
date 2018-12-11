#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import sys
import time
import argparse
import pickle
import numpy as np
import re
import inspect
import time 
import torch
from torchqrnn import QRNN
import torch
from torch import optim
from torch.autograd import Variable
import torch.nn as nn

#from data import get_nli, get_batch, build_vocab
#from mutils import get_optimizer
#from models import NLINet

start_time = time.time()
W2V_PATH = "/home/dc/cs230_project/dataset/GloVe/glove.840B.300d.txt"




parser = argparse.ArgumentParser(description='NLI training')
# paths
parser.add_argument("--nlipath", type=str, default='/home/dc/InferSent/dataset/SNLI', help="NLI data path (SNLI or MultiNLI)")
parser.add_argument("--outputdir", type=str, default='savedir/', help="Output directory")
parser.add_argument("--outputmodelname", type=str, default='3layernonlinear_dropout2.pickle')


# training
parser.add_argument("--n_epochs", type=int, default=30)
parser.add_argument("--batch_size", type=int, default=128)
#this only works if num_layers>1
parser.add_argument("--dpout_model", type=float, default=0., help="encoder dropout")
#this is only for the dropout after batchnorm in nonlinear
parser.add_argument("--dpout_fc", type=float, default=0.2, help="classifier dropout")
parser.add_argument("--nonlinear_fc", type=float, default=1, help="use nonlinearity in fc")
parser.add_argument("--optimizer", type=str, default="sgd,lr=0.1", help="adam or sgd,lr=0.1")
parser.add_argument("--lrshrink", type=float, default=5, help="shrink factor for sgd")
parser.add_argument("--weight_decay", type=float, default=5e-4, help="weight decay for sgd")

parser.add_argument("--decay", type=float, default=0.99, help="lr decay")
parser.add_argument("--minlr", type=float, default=1e-5, help="minimum lr")
parser.add_argument("--max_norm", type=float, default=5., help="max norm (grad clipping)")

# model
parser.add_argument("--encoder_type", type=str, default='InferSent', help="see list of encoders")
parser.add_argument("--enc_lstm_dim", type=int, default=2048, help="encoder nhid dimension")
parser.add_argument("--n_enc_layers", type=int, default=1, help="encoder num layers")
parser.add_argument("--fc_dim", type=int, default=512, help="nhid of fc layers")
parser.add_argument("--n_classes", type=int, default=2, help="duplicate/not duplicate")
parser.add_argument("--pool_type", type=str, default='max', help="max or mean")
parser.add_argument("--word_emb_dim", type=int, default='300', help="embedding dim")
parser.add_argument("--LSTM_num_layers", type=int, default='1', help="LSTM num layers")
parser.add_argument("--data_dir", type=str, default='/home/dc/cs230_project/dataset', help="store duplicate questions")

# gpu
parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
parser.add_argument("--seed", type=int, default=1234, help="seed")


params, _ = parser.parse_known_args()

# set gpu device
torch.cuda.set_device(params.gpu_id)

# print parameters passed, and all parameters
print('\ntogrep : {0}\n'.format(sys.argv[1:]))
print(params)

"""
SEED
"""
np.random.seed(params.seed)
torch.manual_seed(params.seed)
torch.cuda.manual_seed(params.seed)


#data formatting
QUORA_PATH="/home/dc/cs230_project/dataset"

def clean_quora(quora_path):
    '''
    input: path of quora tsv file downloaded from kaggle
    output: df with questions <10 chars removed
    
    '''
    df = pd.read_csv(os.path.join(quora_path,"quora_duplicate_questions.tsv"),sep="\t")
    print(df.head())
    df = df.drop(["id","qid1","qid2"],axis=1)
    print(df.count())
    df=df.dropna()
    print(df.count())
    df['q1_len'] = df['question1'].apply(len)
    df['q2_len'] = df['question2'].apply(len)
    print(df.head())
    #print(df.loc[df['q1_len'] < 10])
    #print(df.loc[df['q2_len'] < 10])
    df = df.loc[ (df['q1_len'] > 10) & (df['q2_len'] > 10)]
    print(df.count())
    return df



def load_single_file(filename):
    fh = open(os.path.join(params.data_dir,filename+'.pkl'),'rb')
    data = pickle.load(fh)
    fh.close()
    return data

def load_data():
    '''
    remove <10 char length question rows. this isnt enough cleaning.
    '''
    X_train = load_single_file("X_train")
    X_valid = load_single_file("X_valid")
    X_test = load_single_file("X_test")
    y_train = load_single_file("y_train")
    y_valid = load_single_file("y_valid")
    y_test = load_single_file("y_test")
    return X_train, X_valid, X_test, y_train,y_valid, y_test
    
def load_data_1sent():
    '''
    1 sentence is same as 0 periods or 1 period? 
    loads X_train_1sent where each question is only 1 sentence for both question 1 and question 2.
    '''
    X_train = load_single_file("X_train_1sent")
    X_valid = load_single_file("X_valid_1sent")
    X_test = load_single_file("X_test_1sent")
    y_train = load_single_file("y_train_1sent")
    y_valid = load_single_file("y_valid_1sent")
    y_test = load_single_file("y_test_1sent")
    return X_train, X_valid, X_test, y_train,y_valid, y_test

def load_data_2sent():
    '''
    has to have 2 periods or more. starts to become a document. do we remove the 0 sentences? 
    then we have very little data so we have to go to a different dataset. 
    loads X_train_2sent which contains data with max of 2 sentences per question
    '''
    X_train = load_single_file("X_train_2sent")
    X_valid = load_single_file("X_valid_2sent")
    X_test = load_single_file("X_test_2sent")
    y_train = load_single_file("y_train_2sent")
    y_valid = load_single_file("y_valid_2sent")
    y_test = load_single_file("y_test_2sent")
    return X_train, X_valid, X_test, y_train,y_valid, y_test
    
    
def format_data(X_train, X_valid,X_test, y_train,y_valid,y_test):
    s1 = {}
    s2 = {}
    target = {}
    s1['train'],s1['dev'],s1['test'],s2['train'],s2['dev'],s2['test'] = {},{},{},{},{},{}
    target['train'],target['dev'],target['test']={},{},{}

    s1['train']['sent'] = [x for x in X_train[:,0]]
    s2['train']['sent'] = [x for x in X_train[:,1]]
    s1['dev']['sent'] = [x for x in X_valid[:,0]]
    s2['dev']['sent'] = [x for x in X_valid[:,1]]
    s1['test']['sent'] = [x for x in X_test[:,0]]
    s2['test']['sent'] = [x for x in X_test[:,1]]
    target['train']['data'] = np.array([x[0] for x in y_train])
    target['dev']['data'] = np.array([x[0] for x in y_valid])
    target['test']['data'] = np.array([x[0] for x in y_test.tolist()])

    train = {'s1': s1['train']['sent'], 's2': s2['train']['sent'],
             'label': target['train']['data']}
    dev = {'s1': s1['dev']['sent'], 's2': s2['dev']['sent'],
           'label': target['dev']['data']}
    test = {'s1': s1['test']['sent'], 's2': s2['test']['sent'],
            'label': target['test']['data']}
    return train,dev,test



def get_word_dict(sentences):
    # create vocab of words
    word_dict = {}
    for sent in sentences:
        for word in sent.split():
            if word not in word_dict:
                word_dict[word] = ''
    word_dict['<s>'] = ''
    word_dict['</s>'] = ''
    word_dict['<p>'] = ''
    return word_dict


def get_glove(word_dict, glove_path):
    # create word_vec with glove vectors
    word_vec = {}
    with open(glove_path) as f:
        for line in f:
            word, vec = line.split(' ', 1)
            if word in word_dict:
                word_vec[word] = np.array(list(map(float, vec.split())))
    print('Found {0}(/{1}) words with glove vectors'.format(
                len(word_vec), len(word_dict)))
    return word_vec


def build_vocab(sentences, glove_path):
    word_dict = get_word_dict(sentences)
    word_vec = get_glove(word_dict, glove_path)
    print('Vocab size : {0}'.format(len(word_vec)))
    return word_vec

def quora():
    X_train,X_valid,X_test,y_train,y_valid,y_test = load_data()
    train,valid,test = format_data(X_train, X_valid,X_test, y_train,y_valid,y_test)
    print(f"quora checkpoint len(train[s1]):{len(train['s1'])},len(train[s2]):{len(train['s2'])},          len(train[label]):{len(train['label'])}")
    print('============')
    print(f"len(valid['s1']):{len(valid['s1'])}, len(valid[s2]):{len(valid['s2'])},           len(valid['label']):{len(valid['label'])}")
    print('============')
    print(f"len(test['s1']):{len(test['s1'])},len(test['s2']):{len(test['s2'])},           len(test['label']):{len(test['label'])}")
          
    word_vec = build_vocab(train['s1'] + train['s2'] +
                       valid['s1'] + valid['s2'] +
                       test['s1'] + test['s2'], W2V_PATH)
    for split in ['s1', 's2']:
        for data_type in ['train', 'valid', 'test']:
            eval(data_type)[split] = np.array([['<s>'] +
            [word for word in sent.split() if word in word_vec] +
            ['</s>'] for sent in eval(data_type)[split]])
    
    return train,valid,test,word_vec

#train, valid, test,word_vec = orig(params,W2V_PATH)
train, valid, test,word_vec = quora()
#print(f"checkpoint after formatting: len(train[s1]):{len(train['s1'])} ,len(train[s2]):{len(train['s2'])} \
#      ,len(train[label]):{len(train['label'])}, len(valid[s2]):{len(valid['s1'])} ,len(valid[s2]):{len(valid['s2'])}, \
#      len(valid[label]):{len(valid['label'])},len(test[s2]):{len(test['s1'])}, len(test[s2]):{len(test['s2'])} \
#      ,len(valid[label]):{len(valid['label'])},len(word_vec):{len(word_vec)}")


"""
MODEL
"""
# model config
config_nli_model = {
    'n_words'        :  300          ,
    'word_emb_dim'   :  params.word_emb_dim   ,
    'enc_lstm_dim'   :  params.enc_lstm_dim   ,
    'n_enc_layers'   :  params.n_enc_layers   ,
    'dpout_model'    :  params.dpout_model    ,
    'dpout_fc'       :  params.dpout_fc       ,
    'fc_dim'         :  params.fc_dim         ,
    'bsize'          :  params.batch_size     ,
    'n_classes'      :  params.n_classes      ,
    'pool_type'      :  params.pool_type      ,
    'nonlinear_fc'   :  params.nonlinear_fc   ,
    'encoder_type'   :  params.encoder_type   ,
    'use_cuda'       :  True                  ,

}


class InferSent(nn.Module):

    def __init__(self, config):
        super(InferSent, self).__init__()
        self.bsize = config['bsize']
        self.word_emb_dim = config['word_emb_dim']
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.pool_type = config['pool_type']
        self.dpout_model = config['dpout_model']
        self.version = 1 if 'version' not in config else config['version']

        self.enc_lstm = nn.LSTM(self.word_emb_dim, self.enc_lstm_dim, params.LSTM_num_layers,
                                bidirectional=True, dropout=self.dpout_model)

        assert self.version in [1, 2]
        if self.version == 1:
            self.bos = '<s>'
            self.eos = '</s>'
            self.max_pad = True
            self.moses_tok = False
        elif self.version == 2:
            self.bos = '<p>'
            self.eos = '</p>'
            self.max_pad = False
            self.moses_tok = True

    def is_cuda(self):
        # either all weights are on cpu or they are on gpu
        return self.enc_lstm.bias_hh_l0.data.is_cuda

    def forward(self, sent_tuple):
        # sent_len: [max_len, ..., min_len] (bsize)
        # sent: Variable(seqlen x bsize x worddim)
        sent, sent_len = sent_tuple

        # Sort by length (keep idx) https://github.com/pytorch/pytorch/issues/3584
        sent_len_sorted, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
        idx_unsort = np.argsort(idx_sort)
        idx_sort = torch.from_numpy(idx_sort).cuda() if self.is_cuda() else torch.from_numpy(idx_sort)
        sent = sent.index_select(1, Variable(idx_sort))

        # Padding perf increase
        sent_packed = nn.utils.rnn.pack_padded_sequence(sent, sent_len_sorted)
        sent_output = self.enc_lstm(sent_packed)[0]  # seqlen x batch x 2*nhid
        sent_output = nn.utils.rnn.pad_packed_sequence(sent_output)[0]

        # Un-sort by length
        idx_unsort = torch.from_numpy(idx_unsort).cuda() if self.is_cuda()             else torch.from_numpy(idx_unsort)
        sent_output = sent_output.index_select(1, Variable(idx_unsort))

        # Pooling
        if self.pool_type == "mean":
            sent_len = Variable(torch.FloatTensor(sent_len.copy())).unsqueeze(1).cuda()
            emb = torch.sum(sent_output, 0).squeeze(0)
            emb = emb / sent_len.expand_as(emb)
        elif self.pool_type == "max":
            if not self.max_pad:
                sent_output[sent_output == 0] = -1e9
            emb = torch.max(sent_output, 0)[0]
            if emb.ndimension() == 3:
                emb = emb.squeeze(0)
                assert emb.ndimension() == 2

        return emb

    def set_w2v_path(self, w2v_path):
        self.w2v_path = w2v_path

    def get_word_dict(self, sentences, tokenize=True):
        # create vocab of words
        word_dict = {}
        sentences = [s.split() if not tokenize else self.tokenize(s) for s in sentences]
        for sent in sentences:
            for word in sent:
                if word not in word_dict:
                    word_dict[word] = ''
        word_dict[self.bos] = ''
        word_dict[self.eos] = ''
        return word_dict

    def get_w2v(self, word_dict):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        # create word_vec with w2v vectors
        word_vec = {}
        with open(self.w2v_path) as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if word in word_dict:
                    word_vec[word] = np.fromstring(vec, sep=' ')
        print('Found %s(/%s) words with w2v vectors' % (len(word_vec), len(word_dict)))
        return word_vec

    def get_w2v_k(self, K):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        # create word_vec with k first w2v vectors
        k = 0
        word_vec = {}
        with open(self.w2v_path) as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if k <= K:
                    word_vec[word] = np.fromstring(vec, sep=' ')
                    k += 1
                if k > K:
                    if word in [self.bos, self.eos]:
                        word_vec[word] = np.fromstring(vec, sep=' ')

                if k > K and all([w in word_vec for w in [self.bos, self.eos]]):
                    break
        return word_vec

    def build_vocab(self, sentences, tokenize=True):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        word_dict = self.get_word_dict(sentences, tokenize)
        self.word_vec = self.get_w2v(word_dict)
        print('Vocab size : %s' % (len(self.word_vec)))

    # build w2v vocab with k most frequent words
    def build_vocab_k_words(self, K):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        self.word_vec = self.get_w2v_k(K)
        print('Vocab size : %s' % (K))

    def update_vocab(self, sentences, tokenize=True):
        assert hasattr(self, 'w2v_path'), 'warning : w2v path not set'
        assert hasattr(self, 'word_vec'), 'build_vocab before updating it'
        word_dict = self.get_word_dict(sentences, tokenize)

        # keep only new words
        for word in self.word_vec:
            if word in word_dict:
                del word_dict[word]

        # udpate vocabulary
        if word_dict:
            new_word_vec = self.get_w2v(word_dict)
            self.word_vec.update(new_word_vec)
        else:
            new_word_vec = []
        print('New vocab size : %s (added %s words)'% (len(self.word_vec), len(new_word_vec)))

    def get_batch(self, batch):
        # sent in batch in decreasing order of lengths
        # batch: (bsize, max_len, word_dim)
        embed = np.zeros((len(batch[0]), len(batch), self.word_emb_dim))

        for i in range(len(batch)):
            for j in range(len(batch[i])):
                embed[j, i, :] = self.word_vec[batch[i][j]]

        return torch.FloatTensor(embed)

    def tokenize(self, s):
        from nltk.tokenize import word_tokenize
        if self.moses_tok:
            s = ' '.join(word_tokenize(s))
            s = s.replace(" n't ", "n 't ")  # HACK to get ~MOSES tokenization
            return s.split()
        else:
            return word_tokenize(s)

    def prepare_samples(self, sentences, bsize, tokenize, verbose):
        sentences = [[self.bos] + s.split() + [self.eos] if not tokenize else
                     [self.bos] + self.tokenize(s) + [self.eos] for s in sentences]
        n_w = np.sum([len(x) for x in sentences])

        # filters words without w2v vectors
        for i in range(len(sentences)):
            s_f = [word for word in sentences[i] if word in self.word_vec]
            if not s_f:
                import warnings
                warnings.warn('No words in "%s" (idx=%s) have w2v vectors.                                Replacing by "</s>"..' % (sentences[i], i))
                s_f = [self.eos]
            sentences[i] = s_f

        lengths = np.array([len(s) for s in sentences])
        n_wk = np.sum(lengths)
        if verbose:
            print('Nb words kept : %s/%s (%.1f%s)' % (
                        n_wk, n_w, 100.0 * n_wk / n_w, '%'))

        # sort by decreasing length
        lengths, idx_sort = np.sort(lengths)[::-1], np.argsort(-lengths)
        sentences = np.array(sentences)[idx_sort]

        return sentences, lengths, idx_sort

    def encode(self, sentences, bsize=64, tokenize=True, verbose=False):
        tic = time.time()
        sentences, lengths, idx_sort = self.prepare_samples(
                        sentences, bsize, tokenize, verbose)

        embeddings = []
        for stidx in range(0, len(sentences), bsize):
            batch = Variable(self.get_batch(
                        sentences[stidx:stidx + bsize]), volatile=True)
            if self.is_cuda():
                batch = batch.cuda()
            batch = self.forward(
                (batch, lengths[stidx:stidx + bsize])).data.cpu().numpy()
            embeddings.append(batch)
        embeddings = np.vstack(embeddings)

        # unsort
        idx_unsort = np.argsort(idx_sort)
        embeddings = embeddings[idx_unsort]

        if verbose:
            print('Speed : %.1f sentences/s (%s mode, bsize=%s)' % (
                    len(embeddings)/(time.time()-tic),
                    'gpu' if self.is_cuda() else 'cpu', bsize))
        return embeddings

    def visualize(self, sent, tokenize=True):

        sent = sent.split() if not tokenize else self.tokenize(sent)
        sent = [[self.bos] + [word for word in sent if word in self.word_vec] + [self.eos]]

        if ' '.join(sent[0]) == '%s %s' % (self.bos, self.eos):
            import warnings
            warnings.warn('No words in "%s" have w2v vectors. Replacing                            by "%s %s"..' % (sent, self.bos, self.eos))
        batch = Variable(self.get_batch(sent), volatile=True)

        if self.is_cuda():
            batch = batch.cuda()
        output = self.enc_lstm(batch)[0]
        output, idxs = torch.max(output, 0)
        # output, idxs = output.squeeze(), idxs.squeeze()
        idxs = idxs.data.cpu().numpy()
        argmaxs = [np.sum((idxs == k)) for k in range(len(sent[0]))]

        # visualize model
        import matplotlib.pyplot as plt
        x = range(len(sent[0]))
        y = [100.0 * n / np.sum(argmaxs) for n in argmaxs]
        plt.xticks(x, sent[0], rotation=45)
        plt.bar(x, y)
        plt.ylabel('%')
        plt.title('Visualisation of words importance')
        plt.show()

        return output, idxs


class NLINet(nn.Module):
    def __init__(self, config):
        super(NLINet, self).__init__()

        # classifier
        self.nonlinear_fc = config['nonlinear_fc']
        self.fc_dim = config['fc_dim']
        self.n_classes = config['n_classes']
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.encoder_type = config['encoder_type']
        self.dpout_fc = config['dpout_fc']

        self.encoder = eval(self.encoder_type)(config)
        self.inputdim = 4*2*self.enc_lstm_dim
        self.inputdim = 4*self.inputdim if self.encoder_type in                         ["ConvNetEncoder", "InnerAttentionMILAEncoder"] else self.inputdim
        self.inputdim = ((int)(self.inputdim/2)) if self.encoder_type == "LSTMEncoder"                                         else self.inputdim
        if self.nonlinear_fc:
            self.classifier = nn.Sequential(
                nn.Linear(self.inputdim, self.fc_dim),
                nn.BatchNorm1d(self.fc_dim),
                nn.ReLU(),
                nn.Dropout(p=self.dpout_fc),
                
                nn.Linear(self.fc_dim, self.fc_dim),
                nn.BatchNorm1d(self.fc_dim),
                nn.ReLU(),
                nn.Dropout(p=self.dpout_fc),
                
                nn.Linear(self.fc_dim, self.fc_dim),
                nn.BatchNorm1d(self.fc_dim),
                nn.ReLU(),
                nn.Dropout(p=self.dpout_fc),
                
                nn.Linear(self.fc_dim, self.n_classes),
                
                )
        else:
            print(f"self.inputdim:{self.inputdim}, self.fc_dim:{self.fc_dim}")
            print(type(self.inputdim),type(self.fc_dim))
            self.classifier = nn.Sequential(
                nn.Linear(self.inputdim, self.fc_dim),
                nn.Linear(self.fc_dim, self.fc_dim),
                nn.Linear(self.fc_dim, self.n_classes)
                )

    def forward(self, s1, s2):
        # s1 : (s1, s1_len)
        u = self.encoder(s1)
        v = self.encoder(s2)

        features = torch.cat((u, v, torch.abs(u-v), u*v), 1)
        output = self.classifier(features)
        return output

    def encode(self, s1):
        emb = self.encoder(s1)
        return emb


def get_optimizer(s):
    """
    Parse optimizer parameters.
    Input should be of the form:
        - "sgd,lr=0.01"
        - "adagrad,lr=0.1,lr_decay=0.05"
    """
    if "," in s:
        method = s[:s.find(',')]
        optim_params = {}
        for x in s[s.find(',') + 1:].split(','):
            split = x.split('=')
            assert len(split) == 2
            assert re.match("^[+-]?(\d+(\.\d*)?|\.\d+)$", split[1]) is not None
            optim_params[split[0]] = float(split[1])
    else:
        method = s
        optim_params = {}

    if method == 'adadelta':
        optim_fn = optim.Adadelta
    elif method == 'adagrad':
        optim_fn = optim.Adagrad
    elif method == 'adam':
        optim_fn = optim.Adam
    elif method == 'adamax':
        optim_fn = optim.Adamax
    elif method == 'asgd':
        optim_fn = optim.ASGD
    elif method == 'rmsprop':
        optim_fn = optim.RMSprop
    elif method == 'rprop':
        optim_fn = optim.Rprop
    elif method == 'sgd':
        optim_fn = optim.SGD
        assert 'lr' in optim_params
    else:
        raise Exception('Unknown optimization method: "%s"' % method)

    # check that we give good parameters to the optimizer
    expected_args = inspect.getargspec(optim_fn.__init__)[0]
    assert expected_args[:2] == ['self', 'params']
    if not all(k in expected_args[2:] for k in optim_params.keys()):
        raise Exception('Unexpected parameters: expected "%s", got "%s"' % (
            str(expected_args[2:]), str(optim_params.keys())))

    return optim_fn, optim_params




# model
encoder_types = ['InferSent', 'BLSTMprojEncoder', 'BGRUlastEncoder',
                 'InnerAttentionMILAEncoder', 'InnerAttentionYANGEncoder',
                 'InnerAttentionNAACLEncoder', 'ConvNetEncoder', 'LSTMEncoder']
assert params.encoder_type in encoder_types, "encoder_type must be in " +                                              str(encoder_types)
nli_net = NLINet(config_nli_model)
print(nli_net)


# loss
weight = torch.FloatTensor(params.n_classes).fill_(1)
#BCE next w2 categories
loss_fn = nn.CrossEntropyLoss(weight=weight)
loss_fn.size_average = False

# optimizer
optim_fn, optim_params = get_optimizer(params.optimizer)
optimizer = optim_fn(nli_net.parameters(), **optim_params)

# cuda by default
nli_net.cuda()
loss_fn.cuda()




"""
TRAIN
"""
val_acc_best = 10
adam_stop = False
stop_training = False
lr = optim_params['lr'] if 'sgd' in params.optimizer else None


def get_batch(batch, word_vec):
    # sent in batch in decreasing order of lengths (bsize, max_len, word_dim)
    lengths = np.array([len(x) for x in batch])
    max_len = np.max(lengths)
    embed = np.zeros((max_len, len(batch), 300))

    for i in range(len(batch)):
        for j in range(len(batch[i])):
            embed[j, i, :] = word_vec[batch[i][j]]

    return torch.from_numpy(embed).float(), lengths



def trainepoch(epoch):
    print('\nTRAINING : Epoch ' + str(epoch))
    nli_net.train()
    all_costs = []
    logs = []
    words_count = 0

    last_time = time.time()
    correct = 0.
    # shuffle the data
    permutation = np.random.permutation(len(train['s1']))
    #print(f"type(permutation):{type(permutation)}")
    #print(f"type(train['s1']):{type(train['s1'])}")
    
    s1 = train['s1'][permutation]
    s2 = train['s2'][permutation]
    target = train['label'][permutation]
    

    optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * params.decay if epoch>1        and 'sgd' in params.optimizer else optimizer.param_groups[0]['lr']
    print('Learning rate : {0}'.format(optimizer.param_groups[0]['lr']))

    for stidx in range(0, len(s1), params.batch_size):
        # prepare batch
        s1_batch, s1_len = get_batch(s1[stidx:stidx + params.batch_size],
                                     word_vec)
        s2_batch, s2_len = get_batch(s2[stidx:stidx + params.batch_size],
                                     word_vec)
        #print(type(s1_batch),type(s2_batch)) #should be list
        #print(f"s1_len:{s1_len},s2_len:{s2_len}")
        s1_batch, s2_batch = Variable(s1_batch.cuda()), Variable(s2_batch.cuda())
        target_batch=target[stidx:stidx + params.batch_size]
        #print(f"target_batch.shape:{target_batch.shape}")
        #print(f"target_batch:{target_batch}")
        #print(f"target shape:{target.shape}")
        #print(f"target:{target[stidx:stidx + params.batch_size]}")
        tgt_batch = Variable(torch.LongTensor(target[stidx:stidx + params.batch_size])).cuda()
        k = s1_batch.size(1)  # actual batch size
        #print(f"tgt_batch:{tgt_batch}")
        #print(f"k:{k}")
        # model forward
        output = nli_net((s1_batch, s1_len), (s2_batch, s2_len))
        #print(f"type(tgt_batch):{type(tgt_batch)}")
        #print(f"type(output):{type(output)}")
        #print(f"output size:{output.size()}")
        
        #print(f"output:{output}")
        #
        pred = output.data.max(1)[1]
        correct += pred.long().eq(tgt_batch.data.long()).cpu().sum()
        assert len(pred) == len(s1[stidx:stidx + params.batch_size])

        # loss
        loss = loss_fn(output, tgt_batch)
        all_costs.append(loss.item())
        words_count += (s1_batch.nelement() + s2_batch.nelement()) / params.word_emb_dim

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient clipping (off by default)
        shrink_factor = 1
        total_norm = 0

        for p in nli_net.parameters():
            if p.requires_grad:
                p.grad.data.div_(k)  # divide by the actual batch size
                total_norm += p.grad.data.norm() ** 2
        total_norm = np.sqrt(total_norm)

        if total_norm > params.max_norm:
            shrink_factor = params.max_norm / total_norm
        current_lr = optimizer.param_groups[0]['lr'] # current lr (no external "lr", for adam)
        optimizer.param_groups[0]['lr'] = current_lr * shrink_factor # just for update

        # optimizer step
        optimizer.step()
        optimizer.param_groups[0]['lr'] = current_lr
        
        if len(all_costs) == 100:
            print(type(correct),correct,correct.item())
            #logs.append('{0} ; loss {1} accuracy:{2} ;'.format(stidx,round(np.mean(all_costs), 2),round(100.*correct.item()/(stidx+k), 2)))
            logs.append('{0} ; loss {1} ; sentence/s {2} ; words/s {3} ; accuracy train : {4}'.format(
                            stidx, round(np.mean(all_costs), 2),
                            int(len(all_costs) * params.batch_size / (time.time() - last_time)),
                            int(words_count * 1.0 / (time.time() - last_time)), 
                            round(100.*correct.item()/(stidx+k), 2)))
            print(logs[-1])
            last_time = time.time()
            words_count = 0
            all_costs = []
    train_acc = round(100 * correct.item()/len(s1), 2)
    print('results : epoch {0} ; mean accuracy train : {1}'
          .format(epoch, train_acc))
    return train_acc


def evaluate(epoch, eval_type='valid', final_eval=False):
    nli_net.eval()
    correct = 0.
    global val_acc_best, lr, stop_training, adam_stop

    if eval_type == 'valid':
        print('\nVALIDATION : Epoch {0}'.format(epoch))

    s1 = valid['s1'] if eval_type == 'valid' else test['s1']
    s2 = valid['s2'] if eval_type == 'valid' else test['s2']
    target = valid['label'] if eval_type == 'valid' else test['label']

    for i in range(0, len(s1), params.batch_size):
        # prepare batch
        s1_batch, s1_len = get_batch(s1[i:i + params.batch_size], word_vec)
        s2_batch, s2_len = get_batch(s2[i:i + params.batch_size], word_vec)
        s1_batch, s2_batch = Variable(s1_batch.cuda()), Variable(s2_batch.cuda())
        tgt_batch = Variable(torch.LongTensor(target[i:i + params.batch_size])).cuda()
        
        # model forward
        output = nli_net((s1_batch, s1_len), (s2_batch, s2_len))

        pred = output.data.max(1)[1]
        correct += pred.long().eq(tgt_batch.data.long()).cpu().sum()

    # save model
    eval_acc = round(100 * correct.item() / len(s1), 2)
    if final_eval:
        print('finalgrep : accuracy {0} : {1}'.format(eval_type, eval_acc))
    else:
        print('togrep : results : epoch {0} ; mean accuracy {1} :              {2}'.format(epoch, eval_type, eval_acc))

    if eval_type == 'valid' and epoch <= params.n_epochs:
        if eval_acc > val_acc_best:
            print('saving model at epoch {0}'.format(epoch))
            if not os.path.exists(params.outputdir):
                os.makedirs(params.outputdir)
            torch.save(nli_net.state_dict(), os.path.join(params.outputdir,
                       params.outputmodelname))
            val_acc_best = eval_acc
        else:
            if 'sgd' in params.optimizer:
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / params.lrshrink
                print('Shrinking lr by : {0}. New lr = {1}'
                      .format(params.lrshrink,
                              optimizer.param_groups[0]['lr']))
                if optimizer.param_groups[0]['lr'] < params.minlr:
                    stop_training = True
            if 'adam' in params.optimizer:
                # early stopping (at 2nd decrease in accuracy)
                stop_training = adam_stop
                adam_stop = True
    return eval_acc


"""
Train model on Natural Language Inference task
"""
epoch = 1
print(f"total num epochs:{params.n_epochs}")

while not stop_training and epoch <= params.n_epochs:
    train_acc = trainepoch(epoch)
    eval_acc = evaluate(epoch, 'valid')
    epoch += 1
    
#nli_net.save_state_dict(os.path.join(params.outputdir, params.outputmodelname))
# Run best model on test set.
#nli_net.load_state_dict(os.path.join(params.outputdir, params.outputmodelname))
print("saving state dict")
torch.save(nli_net.state_dict,os.path.join(params.outputdir, params.outputmodelname + "3layernonlinear_dropout2_fullmodel.pt"))
print("done saving state dict")


print('\nTEST : Epoch {0}'.format(epoch))
print('calculating validation error')
evaluate(1e6, 'valid', True)
print('calculating test error')
evaluate(0, 'test', True)

# Save encoder instead of full model
torch.save(nli_net.encoder.state_dict(), os.path.join(params.outputdir, params.outputmodelname + '.encoder.pkl'))
#save entire model...
elapsed_time = time.time() - start_time

print("fin",elapsed_time)


# In[ ]:





# In[ ]:




