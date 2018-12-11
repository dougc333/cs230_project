#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import argparse
import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser(description='NLI training')
# paths
parser.add_argument("--data_dir", type=str, default='/home/dc/cs230_project/dataset', help="store duplicate questions")
params, _ = parser.parse_known_args()

print(params)

class MakeData():
    def __init__(self):
        '''
        nothing here
        '''
        self.W2V_PATH = "/home/dc/cs230_project/dataset/GloVe/glove.840B.300d.txt"
    
    def save(self,X_train,X_valid,X_test,y_train,y_valid,y_test,
             X_train_fn="X_train",X_valid_fn="X_valid",X_test_fn="X_test", 
             y_train_fn="y_train",y_valid_fn="y_valid",y_test_fn="y_test",small=False,big=False,clean=False):
    
        checkList=[]
        checkList.append(big), checkList.append(small), checkList.append(clean)
        assert(checkList.count(True)==1)
    
        self.save_single_file(X_train_fn,X_train,big=big,small=small,clean=clean)
        self.save_single_file(X_valid_fn,X_valid,big=big,small=small,clean=clean)
        self.save_single_file(X_test_fn,X_test,big=big,small=small,clean=clean)
        self.save_single_file(y_train_fn,y_train,big=big,small=small,clean=clean)
        self.save_single_file(y_valid_fn,y_valid,big=big,small=small,clean=clean)
        self.save_single_file(y_test_fn,y_test,big=big,small=small,clean=clean)
     

    def save_single_file(self,filename,data,big=False,small=False,clean=False):
        '''
        input: filename to be saved, 3 boolean values 
        if big: save data to filename.pkl
        if small: save data to filename_small.pkl
        if clean: save data to filename_clean.pkl
        '''
        #check arguments only 1 can be true
        checkList=[]
        checkList.append(big), checkList.append(small), checkList.append(clean)
        assert(checkList.count(True)==1)
    
        if (big==True):
            fh = open(os.path.join(params.data_dir,filename+'.pkl'), 'wb+')
            pickle.dump(data, fh)
            fh.close()
        elif(small==True):
            fh = open(os.path.join(params.data_dir,filename+'_small'+'.pkl'), 'wb+')
            pickle.dump(data, fh)
            fh.close()
        elif(clean==True):
            fh = open(os.path.join(params.data_dir,filename+'_clean'+'.pkl'), 'wb+')
            pickle.dump(data, fh)
            fh.close()


    def load_single_file(self,filename):
        fh = open(os.path.join(params.data_dir,filename+'.pkl'),'rb')
        data = pickle.load(fh)
        fh.close()
        return data


    def load_data(self,small=False,big=False, clean=False):
        '''
        load data from pkl files
        '''
        checkList=[]
        checkList.append(big), checkList.append(small), checkList.append(clean)
        assert(checkList.count(True)==1)
        
        if small==True:
            print("loading small")
            X_train = self.load_single_file("X_train_small")
            X_valid = self.load_single_file("X_valid_small")
            X_test = self.load_single_file("X_test_small")
            y_train = self.load_single_file("y_train_small")
            y_valid = self.load_single_file("y_valid_small")
            y_test = self.load_single_file("y_test_small")
        elif(big==True):
            print("loading big")
            X_train = self.load_single_file("X_train")
            X_valid = self.load_single_file("X_valid")
            X_test = self.load_single_file("X_test")
            y_train = self.load_single_file("y_train")
            y_valid = self.load_single_file("y_valid")
            y_test = self.load_single_file("y_test")
        elif(clean==True):
            print("loading clean")
            X_train = self.load_single_file("X_train_clean")
            X_valid = self.load_single_file("X_valid_clean")
            X_test = self.load_single_file("X_test_clean")
            y_train = self.load_single_file("y_train_clean")
            y_valid = self.load_single_file("y_valid_clean")
            y_test = self.load_single_file("y_test_clean")
    
        return X_train, X_valid, X_test, y_train,y_valid, y_test


    def num_sent(self,text):
        '''
        this is a gross approximation because we dont tokenize and clearly ellipsis are used in the text
        and they increase the sentence count
        '''
        return text.count('.')

    def num_words(self,text):
        return len(text.split())


    def clean_quora(self,quora_path,big=False,small=False,clean=False):
        '''
        input: path of quora tsv file downloaded from kaggle
        output: df with questions <10 chars removed
        orig: miminmal cleaning just enough to get training to pass. Remove blank lines or questions with no words.
        clean_ten: clean char less than 10 long bc you get things like What??? which clearly are junk and not duplicates
        clean_most: clean long questions which are clearly not duplicates
        '''
        checkList=[]
        checkList.append(big), checkList.append(small), checkList.append(clean)
        assert(checkList.count(True)==1)
    
        df = pd.read_csv(os.path.join(params.data_dir,"quora_duplicate_questions.tsv"),sep="\t")
        #print(df.head())
        df = df.drop(["id","qid1","qid2"],axis=1)
        print(f"before cleaning:{df.count()}")
        #blanks in question1 or question2 columns cause training loop error
        df=df.dropna()
        print(df.count())
        #404287
        #this is num chars!!! wrong!! 
        df['q1_chars'] = df['question1'].apply(len)
        df['q2_chars'] = df['question2'].apply(len)
        df['len_q1'] = df['question1'].apply(self.num_words)
        df['len_q2'] = df['question2'].apply(self.num_words)
        df['diff'] = df['question1'].apply(self.num_words) - df['question2'].apply(self.num_words)
        df['num_q1_sent'] = df['question1'].apply(self.num_sent)
        df['num_q2_sent'] = df['question2'].apply(self.num_sent)
   
        #print(df.head())
        if big==True or small==True:
            df = df.loc[ (df['q1_chars'] > 10) & (df['q2_chars'] > 10)]
        if clean==True:
            df = df.loc[ (df['q1_chars'] > 10) & (df['q2_chars'] > 10)]
            #not correct
            print(f"before clean True:{df.count()}")
            df = df.loc[ (df['diff'] < 10) & (df['diff'] > -10)]
            print(f"after clean True:{df.count()}")
            #add more
        print(f" after clean:{df.count()}")
    
        return df

    def make_dataset(self,path,small=False,clean=False,big=False,small_percent=0.10):
        '''
        input: path: path where quora_duplicate.tsv
        output: train, dev, valid tsv datasets
        '''
        checkList=[]
        checkList.append(big), checkList.append(small), checkList.append(clean)
        assert(checkList.count(True)==1)
        
        
        df = self.clean_quora(params.data_dir,big=big,small=small,clean=clean)
        print(df.values.shape)#(404290, 3)
        if small==True:
            num_rows = df.count()
            print("total num_rows",num_rows.values)
            num_rows = (int)(num_rows.values[0]*.25)
            print("small num_rows",num_rows)
            df = df[:num_rows]
            print(f"after small processing shape:{df.values.shape}") #(404290, 3)
        if (big==True or clean==True):
            print(f"after processing shape:{df.values.shape}") #(404290, 3)
        
        print(f"after processing, sre you small or large? shape:{df.values.shape}") #(404290, 3)
        #drop rows which have 0 in either column. Must be populated wq1 and q2
        #lowercase and split dataframe
        #df.apply(lambda x: x.str.lower() if(x.dtype == 'object') else x)
        #keep capital letters,
        X_train, X_test, y_train, y_test = train_test_split(df[['question1','question2']].values, df[['is_duplicate']].values, test_size=0.40, random_state=42)
        X_test,X_valid,y_test,y_valid = train_test_split(X_test, y_test, test_size=0.50, random_state=42)
        print(f"X_train.shape:{X_train.shape} X_train.shape:{y_train.shape}")
        print(f"X_test.shape:{X_test.shape} y_test.shape:{y_test.shape}")
        print(f"X_valid.shape:{X_valid.shape} y_valid.shape:{y_valid.shape}")
    
        #print(X_train[:6],y_train[:6])
        #print('---------------------')
        #print(X_test[:6],y_test[:6])
        #print('---------------------')
        #print(X_valid[:6],y_valid[:6])
    
        return X_train,X_valid,X_test,y_train,y_valid,y_test

    
    def format_data(self, X_train, X_valid,X_test, y_train,y_valid,y_test):
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
        
    def get_word_dict(self,sentences):
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
    
    def get_glove(self,word_dict, glove_path):
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
    
    
    def build_vocab(self,sentences, glove_path):
        word_dict = self.get_word_dict(sentences)
        word_vec = self.get_glove(word_dict, glove_path)
        print('Vocab size : {0}'.format(len(word_vec)))
        return word_vec

    def quora(self,big=False,small=False,clean=False):
        X_train,X_valid,X_test,y_train,y_valid,y_test = self.load_data(big=big,small=small,clean=clean)
        train,valid,test = self.format_data(X_train, X_valid,X_test, y_train,y_valid,y_test)
        print(f"quora checkpoint len(train[s1]):{len(train['s1'])},len(train[s2]):{len(train['s2'])},          len(train[label]):{len(train['label'])}")
        print('============')
        print(f"len(valid['s1']):{len(valid['s1'])}, len(valid[s2]):{len(valid['s2'])},           len(valid['label']):{len(valid['label'])}")
        print('============')
        print(f"len(test['s1']):{len(test['s1'])},len(test['s2']):{len(test['s2'])},           len(test['label']):{len(test['label'])}")
          
        word_vec = self.build_vocab(train['s1'] + train['s2'] +
                       valid['s1'] + valid['s2'] +
                       test['s1'] + test['s2'], self.W2V_PATH)
        for split in ['s1', 's2']:
            for data_type in ['train', 'valid', 'test']:
                eval(data_type)[split] = np.array([['<s>'] +
                [word for word in sent.split() if word in word_vec] +
                ['</s>'] for sent in eval(data_type)[split]])
    
        return train,valid,test,word_vec
    
    
    def test(self):
        '''
        create 3 datasets, big, small and clean. Print out stats to verify
        '''
        # test create large dataset
        X_train,X_valid,X_test,y_train,y_valid,y_test = self.make_dataset(params.data_dir, small=False,big=True,clean=False)
        self.save(X_train,X_valid,X_test,y_train,y_valid,y_test,big=True,small=False,clean=False)

        #test loading large dataset
        X_train,X_valid,X_test,y_train,y_valid,y_test = self.load_data(big=True,small=False,clean=False)
        print("should see large dataset: 404158,242494,80832")
        print(type(X_train),X_train.shape,type(y_train),y_train.shape)
        print(type(X_valid),X_valid.shape,type(y_valid),y_valid.shape)
        print(type(X_test),X_test.shape,type(y_test),y_test.shape)

        #test creating small dataset
        X_train_small,X_valid_small,X_test_small,y_train_small,y_valid_small,y_test_small         = self.make_dataset(params.data_dir,big=False,small=True,clean=False)
        self.save(X_train_small,X_valid_small,X_test_small,y_train_small,y_valid_small,y_test_small,
             big=False,small=True,clean=False)

        #test loading small dataset
        X_train,X_valid,X_test,y_train,y_valid,y_test = self.load_data(small=True,big=False,clean=False)
        print("should see small dataset: 60623,20208")
        print(type(X_train),X_train.shape,type(y_train),y_train.shape)
        print(type(X_valid),X_valid.shape,type(y_valid),y_valid.shape)
        print(type(X_test),X_test.shape,type(y_test),y_test.shape)

        #test creating clean dataset
        X_train_clean,X_valid_clean,X_test_clean,y_train_clean,y_valid_clean,y_test_clean =         self.make_dataset(params.data_dir, small=False,big=False,clean=True)
        self.save(X_train_clean,X_valid_clean,X_test_clean,y_train_clean,y_valid_clean,y_test_clean, 
             big=False,small=False,clean=True)
        #test loading clean dataset
        X_train,X_valid,X_test,y_train,y_valid,y_test = self.load_data(small=False,big=False,clean=True)
        print("should see clean dataset: 404158,80832")
        print(type(X_train),X_train.shape,type(y_train),y_train.shape)
        print(type(X_valid),X_valid.shape,type(y_valid),y_valid.shape)
        print(type(X_test),X_test.shape,type(y_test),y_test.shape)


# In[2]:


#make_data = MakeData()
#make_data.test()


# In[ ]:




