
# coding: utf-8

# In[2]:



import pandas as pd
import re
import csv
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
from nltk.tokenize import TweetTokenizer
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import model_selection
from sklearn.ensemble import AdaBoostClassifier
from allennlp.commands.elmo import ElmoEmbedder
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import os
import sys
import numpy as  np
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support

stop_words = set(stopwords.words('english'))


# In[3]:


def prepare_data(filename):
    tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)
    ps=PorterStemmer()
    r_anum = re.compile(r'([^\sa-z0-9.(?)!])+')
    r_white = re.compile(r'[\s.(?)!]+')
    text = []; label_lists = [];
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            post = str(row[1])
            post=tknzr.tokenize(post)
            post_preproc = [ps.stem(w) for w in post if not w.strip().lower() in stop_words]
            post_preproc= str(' '.join(post_preproc)).lower()
            row_clean = r_white.sub(' ', r_anum.sub('', post_preproc.lower())).strip()
            text.append(row_clean)
            if row[2] == "NOT":
                label_lists.append(0)
            else:
                label_lists.append(1)
    return text, label_lists


def prepare_data1(filename):
    tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)
    ps=PorterStemmer()
    r_anum = re.compile(r'([^\sa-z0-9.(?)!])+')
    r_white = re.compile(r'[\s.(?)!]+')
    text = [];
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter = '\t')
        for row in reader:
            post = str(row[0])
            post=tknzr.tokenize(post)
            post_preproc = [ps.stem(w) for w in post if not w.strip().lower() in stop_words]
            post_preproc= str(' '.join(post_preproc)).lower()
            row_clean = r_white.sub(' ', r_anum.sub('', post_preproc.lower())).strip()
            text.append(row_clean)
    return text

# In[37]:


def get_model(m_type):
    if m_type == 'logistic regression':
        logreg = LogisticRegression()
    elif m_type == "random_forest":
        logreg = RandomForestClassifier(n_estimators=100, n_jobs=-1, class_weight="balanced")
    elif m_type == "svm":
        logreg = LinearSVC(C=1.0,class_weight = "balanced")
    elif m_type == "GBT":
        logreg = GradientBoostingClassifier(n_estimators=50)
    elif m_type == "Adaboost":
        logreg = AdaBoostClassifier(n_estimators=50) 
    else:
        print ("ERROR: Please specify a correst model")
        return None
    return logreg


# In[38]:


def get_embedding_weights(filename, sep):
    embed_dict = {}
    file = open(filename,'r')
    for line in file.readlines():
        row = line.strip().split(sep)
        embed_dict[row[0]] = row[1:]
    print('Loaded from file: ' + str(filename))
    file.close()
    return embed_dict


# In[39]:


def get_features_glove(Tdata,emb):
    features = []
    tknzr = TweetTokenizer()
    for i in range(len(Tdata)):
        concat = np.zeros(300)
        Tdata[i] = Tdata[i].lower()
        text = ''.join([c for c in Tdata[i] if c not in punctuation])
        tok = tknzr.tokenize(text)
        toklen = 1
        for wor in range(len(tok)):
            if tok[wor] in emb:
                toklen += 1
                flist = [float(i) for i in emb[str(tok[wor])]]
                concat= flist + concat
        concat = concat/toklen
        features.append(concat)
    return features


# In[40]:


def elmo_save(data, dir_filepath, emb_size,elmo):   
        for ind_sen, sent in enumerate(data):
            w_ind = 0
            word_len_post = len(data[ind_sen].split(' '))
            feats_ID = np.zeros((word_len_post, emb_size))
            sent_words = sent.split(' ')
            vectors = elmo.embed_sentence(sent_words)
            for i in range(len(sent_words)):
                feats_ID[w_ind] = np.concatenate((vectors[0][i], vectors[1][i], vectors[2][i]), axis=None)
                w_ind += 1
            np.save(dir_filepath + str(ind_sen) + '.npy', feats_ID)


# In[41]:


def extract_feature(features,X_train, X_test, y_train):
    
    if features == 'tfidf_word':
        tfidf_transformer = TfidfTransformer(norm = 'l2')
        count_vec = CountVectorizer(analyzer="word",max_features = 5000,stop_words='english',ngram_range = (1,2))
        bow_transformer_train = count_vec.fit_transform(X_train)
        bow_transformer_test =count_vec.transform(X_test)
        train_features = tfidf_transformer.fit_transform(bow_transformer_train)
        test_features= tfidf_transformer.transform(bow_transformer_test )
    elif features == 'tfidf_char':
        tfidf_transformer = TfidfTransformer(norm = 'l2')
        count_vec = CountVectorizer(analyzer="char",max_features = 5000, ngram_range = (1,5))
        bow_transformer_train = count_vec.fit_transform(X_train)
        bow_transformer_test =count_vec.transform(X_test)
        train_features = tfidf_transformer.fit_transform(bow_transformer_train)
        test_features= tfidf_transformer.transform(bow_transformer_test ) 
    elif features == 'glove':
        filename_glove = '/home/varsha/hasoc/glovetwitter/glovetwitter200d.txt'
        emb = get_embedding_weights(filename_glove, ' ')
        train_features = get_features_glove(X_train,emb)
        test_features = get_features_glove(X_test,emb)
    elif features == 'elmo':
        train_features=[]
        test_features =[]
        elmo_save_filepath_train = 'elmo/train/' 
        elmo_save_filepath_test = 'elmo/test/'
        if not os.path.isfile(elmo_save_filepath_train + '0.npy') or  os.path.isfile(elmo_save_filepath_test + '0.npy'):
            os.makedirs(elmo_save_filepath_train, exist_ok=True)
            os.makedirs(elmo_save_filepath_test, exist_ok=True)
            elmo = ElmoEmbedder()
            elmo_save(X_train,elmo_save_filepath_train, 3072,elmo)
            elmo_save(X_test,elmo_save_filepath_test, 3072,elmo)
        for i in range(len(X_train)):
            arr = np.load(elmo_save_filepath_train + str(i) + '.npy')
            avg_words = np.mean(arr, axis=0)
            train_features.append(avg_words)
        #train_features = np.asarray(train_features)
        for i in range(len(X_test)):
            arr = np.load(elmo_save_filepath_test+str(i) + '.npy')
            avg_words = np.mean(arr, axis=0)
            test_features.append(avg_words)
        #test_features = np.asarray(test_features)
        
    return train_features, test_features


# In[42]:


def classification_model(X_train, X_test, y_train,model_type):
    model = get_model(model_type)
    # print(X_train[:10], y_train[:10])
    # X_test = X_train[1500:2000]
    # y_test = y_train[1500:2000]
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    y_pred_1 = []
    for i in y_pred:
        if i==1:
            y_pred_1.append('HOF')
        else:
            y_pred_1.append('NOT')
    # for i in y_pred:
    #     print(i)
    df = pd.DataFrame(y_pred_1)

    df.to_csv('submission.csv',index=None)
    # acc = f1_score(y_test, y_pred, average='weighted')
    # print (acc, y_pred)
    # y_score = model.decision_function(X_test)
    #average_precision=average_precision_score(y_test,y_score)
    #recall_score(y_true,y_pred,average='macro')
    #print (average_precision)
    # ans=precision_recall_fscore_support(y_tested, y_pred, average='binary')
    # print (ans)


# In[43]:


X_train, y_train = prepare_data('./data/train.csv')
X_test = prepare_data1('./data/validation.csv')
# X_train, X_test, y_train, y_test = train_test_split(text,labels, test_size=0.20, random_state=42)
train_data , test_data = extract_feature('tfidf_word', X_train, X_test, y_train)
classification_model(train_data ,test_data, y_train,'Adaboost')
// Added
