# -*- coding: utf-8 -*-

from google.colab import drive
drive.mount('/content/gdrive')

import numpy as np
import pandas as pd
from tweetokenize import Tokenizer
import emojis
import re
import wordsegment as ws

df=pd.read_csv("/home/singular/Documents/dataset/data_train.csv")

UNICODE_EMOJI = {v: k for k, v in emojis.EMOJI_UNICODE.items()}

REV_UNICODE_EMOJI = {}

for a,b in emojis.EMOJI_UNICODE.items():
    a=a.replace(':','').replace('_',' ').replace('-','')
    REV_UNICODE_EMOJI[b]=a

mispell_dict = {"tukde":"break","nfuck":"fuck","antiamerica":"anti america","bhikhari":"begger","islamicterrorists":"terrorists","balidan":"badge","balidaan":"badge","naali":"drain","hearnoevil":"hear no evil","nodeal":"no deal","trumperrhoids":"haemorrhoids","motherfuckerr":"motherfucker","balidhan":"badge","banggaan":"fuck","nfact":"fact","despiteof":"despite of","fartman":"fart man","placcards":"play cards","betichod":"fuck","nagain":"snake","hypocrats":"hypocrite","betwern":"between","pissful":"piss","theasshole":"the asshole","balidhaan":"badge","atankwad":"terrorist","trumperrhoid":"haemorrhoids",}
punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', }
contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have" }

import string
lst=[]
no_lst=[]
ws.load()

def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True
    
def Clean(text):
    gettokens = Tokenizer(usernames="",urls="",numbers="")
    t=gettokens.tokenize(text)
    tt=[]

    for i in range(len(t)):
        if t[i] in UNICODE_EMOJI:
            t[i]=''
            # t[i]=REV_UNICODE_EMOJI[t[i]]
            
    for i in range(len(t)):
        if(isEnglish(t[i])):
            tt.append(t[i].lower())
            
    return ' '.join(tt).encode('utf-8')

for i in range(len(df)):
    df.at[i,"text"]=Clean(df["text"][i])
    
def ExpandHashTags(text):
    t=text.split(" ")
    tt=[]
    
    for i in range(len(t)):
        if '#' in t[i]:
            tag=t[i][1:].lower()
            filter_tag=re.sub('\d', '', tag)
            if(len(filter_tag)==0):
                continue
            tt.append(' '.join(ws.segment(filter_tag)))
        else:
            tt.append(t[i])
            
    return ' '.join(tt).encode('utf-8')

# for i in range(len(df)):
#     df.at[i,"text"]=ExpandHashTags(df["text"][i])

def clean_contractions(text, mapping):
  specials = ["’", "‘", "´", "`"]
  for s in specials:
      text = text.replace(s, "'")
  text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])
  return text

df['text'] = df['text'].apply(lambda x: clean_contractions(x, contraction_mapping))

def clean_special_chars(text, punct, mapping):
  for p in mapping:
    text = text.replace(p, mapping[p])
  
  for p in punct:
    text = text.replace(p,'')
  
  specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}  # Other special characters that I have to deal with in last
  for s in specials:
    text = text.replace(s, specials[s])
  
  return text

df['text'] = df['text'].apply(lambda x: clean_special_chars(x, punct, punct_mapping))

def correct_spelling(x, dic):
    for word in dic.keys():
        x = x.replace(word, dic[word])
    return x

df['text'] = df['text'].apply(lambda x: correct_spelling(x, mispell_dict))

df.to_csv("hate_train_2.csv")

df=pd.read_csv("/home/singular/Documents/dataset/data_test.csv")

UNICODE_EMOJI = {v: k for k, v in emojis.EMOJI_UNICODE.items()}

REV_UNICODE_EMOJI = {}

for a,b in emojis.EMOJI_UNICODE.items():
    a=a.replace(':','').replace('_',' ').replace('-','')
    REV_UNICODE_EMOJI[b]=a

mispell_dict = {"tukde":"break","nfuck":"fuck","antiamerica":"anti america","bhikhari":"begger","islamicterrorists":"terrorists","balidan":"badge","balidaan":"badge","naali":"drain","hearnoevil":"hear no evil","nodeal":"no deal","trumperrhoids":"haemorrhoids","motherfuckerr":"motherfucker","balidhan":"badge","banggaan":"fuck","nfact":"fact","despiteof":"despite of","fartman":"fart man","placcards":"play cards","betichod":"fuck","nagain":"snake","hypocrats":"hypocrite","betwern":"between","pissful":"piss","theasshole":"the asshole","balidhaan":"badge","atankwad":"terrorist","trumperrhoid":"haemorrhoids",}
punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', }
contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have" }

import string
lst=[]
no_lst=[]
ws.load()

def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True
    
def Clean(text):
    gettokens = Tokenizer(usernames="",urls="",numbers="")
    t=gettokens.tokenize(text)
    tt=[]

    for i in range(len(t)):
        if t[i] in UNICODE_EMOJI:
            t[i]=''
            # t[i]=REV_UNICODE_EMOJI[t[i]]
            
    for i in range(len(t)):
        if(isEnglish(t[i])):
            tt.append(t[i].lower())
            
    return ' '.join(tt).encode('utf-8')

for i in range(len(df)):
    df.at[i,"text"]=Clean(df["text"][i])
    
def ExpandHashTags(text):
    t=text.split(" ")
    tt=[]
    
    for i in range(len(t)):
        if '#' in t[i]:
            tag=t[i][1:].lower()
            filter_tag=re.sub('\d', '', tag)
            if(len(filter_tag)==0):
                continue
            tt.append(' '.join(ws.segment(filter_tag)))
        else:
            tt.append(t[i])
            
    return ' '.join(tt).encode('utf-8')

# for i in range(len(df)):
#     df.at[i,"text"]=ExpandHashTags(df["text"][i])

def clean_contractions(text, mapping):
  specials = ["’", "‘", "´", "`"]
  for s in specials:
      text = text.replace(s, "'")
  text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])
  return text

df['text'] = df['text'].apply(lambda x: clean_contractions(x, contraction_mapping))

def clean_special_chars(text, punct, mapping):
  for p in mapping:
    text = text.replace(p, mapping[p])
  
  for p in punct:
    text = text.replace(p,'')
  
  specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}  # Other special characters that I have to deal with in last
  for s in specials:
    text = text.replace(s, specials[s])
  
  return text

df['text'] = df['text'].apply(lambda x: clean_special_chars(x, punct, punct_mapping))

def correct_spelling(x, dic):
    for word in dic.keys():
        x = x.replace(word, dic[word])
    return x

df['text'] = df['text'].apply(lambda x: correct_spelling(x, mispell_dict))

df.to_csv("hate_test_2.csv")

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from statistics import mean,stdev

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC,LinearSVC
from sklearn.metrics import accuracy_score,f1_score,classification_report
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from mlxtend.regressor import StackingRegressor
from xgboost import XGBClassifier

from sklearn.model_selection import validation_curve

import lightgbm as lgb

import nltk
nltk.download('wordnet')
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
from nltk.corpus import stopwords
nltk.download('stopwords')


df_train=pd.read_csv("/content/gdrive/My Drive/data/deep_train_2.csv")
df_train.drop("Unnamed: 0",axis=1,inplace=True)
df_test=pd.read_csv("/content/gdrive/My Drive/data/deep_test_2.csv")
df_test.drop("Unnamed: 0",axis=1,inplace=True)

df_final["text"].fillna("",inplace=True)

def Clean(text):
  lemmatizer = WordNetLemmatizer() 
  words=text.split(" ")
  new_sen=[]

  for w in words:
    new_sen.append(lemmatizer.lemmatize(w))

  stop_words = set(stopwords.words('english')) 
  # return " ".join([w for w in new_sen if not w in stop_words])
  return " ".join(new_sen)

for i in range(len(df_train)):
  df_train.at[i,"text"]=Clean(df_train["text"][i])

for i in range(len(df_test)):
  df_test.at[i,"text"]=Clean(df_test["text"][i])

train_x=df_train["text"].values
test_x=df_test["text"].values

train_y=df_train["labels"].values

def DisplayResult(actual,pred):
  print("Accuracy = {}".format(accuracy_score(actual,pred)))
  print("F1(Macro) = {}".format(f1_score(actual,pred,average="macro")))
  print("F1(Default) = {}".format(f1_score(actual,pred)))
  print("F1(Weighted) = {}".format(f1_score(actual,pred,average="weighted")))
  print(classification_report(actual,pred))

def GetResult(actual,pred):
  return [accuracy_score(actual,pred),f1_score(actual,pred)]

c3=XGBClassifier(min_child_weight=5)
c4=lgb.LGBMClassifier(num_leaves=5,learning_rate=0.05, n_estimators=100,max_bin = 45, bagging_fraction = 0.9,bagging_freq = 5, 
                      feature_fraction = 0.1,feature_fraction_seed=5, bagging_seed=9,min_data_in_leaf =6, min_sum_hessian_in_leaf = 5)

c5=RandomForestClassifier(n_estimators=1200,max_depth=50,min_samples_leaf=3,min_samples_split=10,max_leaf_nodes=80,max_samples=0.7))
c6=lgb.LGBMClassifier()
full_model = StackingRegressor(regressors=[c4,c5],meta_regressor=c3)

pipe2=Pipeline([('vect',TfidfVectorizer(ngram_range=(1, 1), min_df=1, use_idf=True, smooth_idf=True)),('clf',full_model),])

pipe2.fit(train_x,train_y)
y_test_pred=list(pipe2.predict(test_x))

df=pd.DataFrame(y_test_pred,columns=["labels"])
