
import pandas as pd
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

import pandas, xgboost, numpy, textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers

Corpus=pd.read_csv('train.csv')

Corpus.head()

import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

np.random.seed(500)

import nltk
nltk.download('wordnet')

from matplotlib import pyplot as plt
plt.hist(Corpus['labels'])

# split the dataset into training and validation datasets 
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(Corpus['text'], Corpus['labels'])

# label encode the target variable 
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)

# create a count vectorizer object 
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(Corpus['text'])

# transform the training and validation data using count vectorizer object
xtrain_count =  count_vect.transform(train_x)
xvalid_count =  count_vect.transform(valid_x)

Corpus_new=pd.read_csv('test.csv')

# word level tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(Corpus['text'])
xtrain_tfidf =  tfidf_vect.transform(train_x)
xvalid_tfidf =  tfidf_vect.transform(valid_x)

xtest_tfidf =  tfidf_vect.transform(Corpus_new.text)

# ngram level tf-idf 
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram.fit(Corpus['text'])
xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)

# characters level tf-idf
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram_chars.fit(Corpus['text'])
xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_x) 
xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(valid_x)

# load the pre-trained word-embedding vectors 
embeddings_index = {}
for i, line in enumerate(open('wiki-news-300d-1M.vec')):
    values = line.split()
    embeddings_index[values[0]] = numpy.asarray(values[1:], dtype='float32')

# create a tokenizer 
token = text.Tokenizer()
token.fit_on_texts(Corpus['text'])
word_index = token.word_index

# convert text to sequence of tokens and pad them to ensure equal length vectors 
train_seq_x = sequence.pad_sequences(token.texts_to_sequences(train_x), maxlen=70)
valid_seq_x = sequence.pad_sequences(token.texts_to_sequences(valid_x), maxlen=70)

# create token-embedding mapping
embedding_matrix = numpy.zeros((len(word_index) + 1, 300))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

#!wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip

#!unzip wiki-news-300d-1M.vec.zip

Corpus['char_count'] = Corpus['text'].apply(len)
Corpus['word_count'] = Corpus['text'].apply(lambda x: len(x.split()))
Corpus['word_density'] = Corpus['char_count'] / (Corpus['word_count']+1)
Corpus['punctuation_count'] = Corpus['text'].apply(lambda x: len("".join(_ for _ in x if _ in string.punctuation))) 
Corpus['title_word_count'] = Corpus['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.istitle()]))
Corpus['upper_case_word_count'] = Corpus['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))

Corpus_new['char_count'] = Corpus_new['text'].apply(len)
Corpus_new['word_count'] = Corpus_new['text'].apply(lambda x: len(x.split()))
Corpus_new['word_density'] = Corpus_new['char_count'] / (Corpus_new['word_count']+1)
Corpus_new['punctuation_count'] = Corpus_new['text'].apply(lambda x: len("".join(_ for _ in x if _ in string.punctuation))) 
Corpus_new['title_word_count'] = Corpus_new['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.istitle()]))
Corpus_new['upper_case_word_count'] = Corpus_new['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))

pos_family = {
    'noun' : ['NN','NNS','NNP','NNPS'],
    'pron' : ['PRP','PRP$','WP','WP$'],
    'verb' : ['VB','VBD','VBG','VBN','VBP','VBZ'],
    'adj' :  ['JJ','JJR','JJS'],
    'adv' : ['RB','RBR','RBS','WRB']
}

# function to check and get the part of speech tag count of a words in a given sentence
def check_pos_tag(x, flag):
    cnt = 0
    try:
        wiki = textblob.TextBlob(x)
        for tup in wiki.tags:
            ppo = list(tup)[1]
            if ppo in pos_family[flag]:
                cnt += 1
    except:
        pass
    return cnt

Corpus['noun_count'] = Corpus['text'].apply(lambda x: check_pos_tag(x, 'noun'))
Corpus['verb_count'] = Corpus['text'].apply(lambda x: check_pos_tag(x, 'verb'))
Corpus['adj_count'] = Corpus['text'].apply(lambda x: check_pos_tag(x, 'adj'))
Corpus['adv_count'] = Corpus['text'].apply(lambda x: check_pos_tag(x, 'adv'))
Corpus['pron_count'] = Corpus['text'].apply(lambda x: check_pos_tag(x, 'pron'))


Corpus_new['noun_count'] = Corpus_new['text'].apply(lambda x: check_pos_tag(x, 'noun'))
Corpus_new['verb_count'] = Corpus_new['text'].apply(lambda x: check_pos_tag(x, 'verb'))
Corpus_new['adj_count'] = Corpus_new['text'].apply(lambda x: check_pos_tag(x, 'adj'))
Corpus_new['adv_count'] = Corpus_new['text'].apply(lambda x: check_pos_tag(x, 'adv'))
Corpus_new['pron_count'] = Corpus_new['text'].apply(lambda x: check_pos_tag(x, 'pron'))

# train a LDA Model
lda_model = decomposition.LatentDirichletAllocation(n_components=20, learning_method='online', max_iter=20)
X_topics = lda_model.fit_transform(xtrain_count)
topic_word = lda_model.components_ 
vocab = count_vect.get_feature_names()

# view the topic models
n_top_words = 10
topic_summaries = []
for i, topic_dist in enumerate(topic_word):
    topic_words = numpy.array(vocab)[numpy.argsort(topic_dist)][:-(n_top_words+1):-1]
    topic_summaries.append(' '.join(topic_words))

def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    # fit the training dataset on the classifier
    if is_neural_net:
      classifier.fit(feature_vector_train, label, epochs=1)
    else:
      classifier.fit(feature_vector_train, label)
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
        #predictions = [int(round(p[0])) for p in predictions]
    
    return metrics.accuracy_score(predictions, valid_y)

classifier=ensemble.RandomForestClassifier()
classifier.fit(xtrain_tfidf, train_y)
predictions = classifier.predict(xvalid_tfidf)
metrics.accuracy_score(predictions, valid_y)

predictions = classifier.predict(xtest_tfidf)

Corpus_new['labels']=predictions

sample_submission = pd.read_csv('sample_submission.csv')
# replacing the label with prediction
sample_submission['labels'] = predictions
sample_submission.head()
# saving the file
sample_submission.to_csv('submission.csv', index=False)

# RF on Count Vectors
accuracy = train_model(ensemble.RandomForestClassifier(), xtrain_count, train_y, xvalid_count)
print( "RF, Count Vectors: ", accuracy)

# RF on Word Level TF IDF Vectors
accuracy = train_model(ensemble.RandomForestClassifier(), xtrain_tfidf, train_y, xvalid_tfidf)
print ("RF, WordLevel TF-IDF: ", accuracy)



# Extereme Gradient Boosting on Count Vectors
accuracy = train_model(xgboost.XGBClassifier(), xtrain_count.tocsc(), train_y, xvalid_count.tocsc())
print( "Xgb, Count Vectors: ", accuracy)

# Extereme Gradient Boosting on Word Level TF IDF Vectors
accuracy = train_model(xgboost.XGBClassifier(), xtrain_tfidf.tocsc(), train_y, xvalid_tfidf.tocsc())
print ("Xgb, WordLevel TF-IDF: ", accuracy)

# Extereme Gradient Boosting on Character Level TF IDF Vectors
accuracy = train_model(xgboost.XGBClassifier(), xtrain_tfidf_ngram_chars.tocsc(), train_y, xvalid_tfidf_ngram_chars.tocsc())
print( "Xgb, CharLevel Vectors: ", accuracy)

def create_rcnn():
      # Add an Input Layer
    input_layer = layers.Input((70, ))

    # Add the word embedding Layer
    embedding_layer = layers.Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], trainable=False)(input_layer)
    embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)
    
    # Add the recurrent layer
    rnn_layer = layers.Bidirectional(layers.GRU(50, return_sequences=True))(embedding_layer)
    
    # Add the convolutional Layer
    conv_layer = layers.Convolution1D(100, 3, activation="relu")(embedding_layer)

    # Add the pooling Layer
    pooling_layer = layers.GlobalMaxPool1D()(conv_layer)

    # Add the output Layers
    output_layer1 = layers.Dense(50, activation="relu")(pooling_layer)
    output_layer1 = layers.Dropout(0.25)(output_layer1)
    output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)

    # Compile the model
    model = models.Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')
    
    return model

classifier = create_rcnn()
accuracy = train_model(classifier, train_seq_x, train_y, valid_seq_x, is_neural_net=True)
print ("CNN, Word Embeddings",  accuracy)

"""# **No Word2vec**"""

text_corpus=pd.read_csv('train.csv')

text_corpus.head()

X=text_corpus.text
y=text_corpus.labels

test_corpus=pd.read_csv('/content/test.csv')
X_Unseen=test_corpus.text

from sklearn.feature_extraction.text import TfidfVectorizer 
 
tfidf_vectorizer=TfidfVectorizer(use_idf=True)
 
# just send in all your docs here
fitted_vectorizer=tfidf_vectorizer.fit(X)
tfidf_vectorizer_vectors=fitted_vectorizer.transform(X)

tfidf_vectorizer_vectors_Unseen=fitted_vectorizer.transform(X_Unseen)

# split the dataset into training and validation datasets 
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(tfidf_vectorizer_vectors,y,test_size=0.3)

from sklearn import datasets, svm, metrics

from sklearn.svm import SVC
clf = SVC(C=10,kernel='rbf')
clf.fit(train_x, train_y)
predicted = clf.predict(valid_x)
metrics.f1_score(valid_y, predicted),metrics.accuracy_score(valid_y, predicted)

from sklearn import ensemble 
from sklearn import metrics
classifier=ensemble.RandomForestClassifier()
classifier.fit(train_x, train_y)
    
# predict the labels on validation dataset
predictions = classifier.predict(valid_x)
metrics.f1_score(valid_y, predictions),metrics.accuracy_score(valid_y, predictions)

classifier=ensemble.RandomForestClassifier(class_weight='balanced_subsample')
classifier.fit(train_x, train_y)
    
# predict the labels on validation dataset
predictions = classifier.predict(valid_x)
metrics.f1_score(valid_y, predictions),metrics.accuracy_score(valid_y, predictions)

from sklearn.ensemble import AdaBoostClassifier
classifier = AdaBoostClassifier(random_state=1)
classifier.fit(train_x, train_y)

# predict the labels on validation dataset
predictions = classifier.predict(valid_x)
print(metrics.f1_score(valid_y, predictions),metrics.accuracy_score(valid_y, predictions))

from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier

classifier = BaggingClassifier(base_estimator=SVC(),
                        n_estimators=100, random_state=0)
classifier.fit(train_x, train_y)

# predict the labels on validation dataset
predictions = classifier.predict(valid_x)
print(metrics.f1_score(valid_y, predictions),metrics.accuracy_score(valid_y, predictions))

from sklearn.ensemble import GradientBoostingClassifier
classifier= GradientBoostingClassifier(learning_rate=0.01,random_state=1)
classifier.fit(train_x, train_y)

# predict the labels on validation dataset
predictions = classifier.predict(valid_x)
print(metrics.f1_score(valid_y, predictions),metrics.accuracy_score(valid_y, predictions))

import xgboost as xgb
classifier=xgb.XGBClassifier(random_state=1,learning_rate=0.01)
classifier.fit(train_x, train_y)

# predict the labels on validation dataset
predictions = classifier.predict(valid_x)
print(metrics.f1_score(valid_y, predictions),metrics.accuracy_score(valid_y, predictions))

import lightgbm as ltb

# fit a lightGBM model to the data
lgbmclassifier = ltb.LGBMClassifier(n_estimators=500,silent=False)
lgbmclassifier.fit(train_x, train_y)

# predict the labels on validation dataset
predictions = lgbmclassifier.predict(valid_x)
print(metrics.f1_score(valid_y, predictions),metrics.accuracy_score(valid_y, predictions))

!pip3 install catboost

from sklearn.metrics import accuracy_score, f1_score
def evaluation(model, X_test, y_test):
    prediction = model.predict(X_test)
    acc = accuracy_score(y_test, prediction)
    f1 = f1_score(y_test, prediction)
   
    print('Acc score:', round(acc, 4))
    print('F1 score:', round(f1,4))

from catboost import CatBoostClassifier
catboost_clf=CatBoostClassifier(l2_leaf_reg=10,loss_function='CrossEntropy',iterations=1000)
catboost_clf.fit(train_x, train_y)

print('Training Accuracy')
evaluation(catboost_clf, train_x, train_y)
print('\nTesting Accuracy')
evaluation(catboost_clf, valid_x, valid_y)

"""# **Tuning CatBoost**"""

model_params = {'depth':[3,1,2,6,4,5,7,8,9,10],
          'iterations':[250,100,500,1000],
          'learning_rate':[0.03,0.001,0.01,0.1,0.2,0.3], 
          'l2_leaf_reg':[3,1,5,10,100],
          'border_count':[32,5,10,20,50,100,200]
          }

model_params = {'depth':[3,1,2,6,4,5],
          'iterations':[1000],
          'learning_rate':[0.001,0.01,0.1,0.2], 
          'l2_leaf_reg':[3,1,5,10,100],
          'border_count':[32,5,10,20]
          }

from sklearn.model_selection import RandomizedSearchCV
# create CatBoostClassifier classifier model

catboost_clf=CatBoostClassifier()
# set up random search meta-estimator
# this will train 100 models over 5 folds of cross validation (500 models total)
clf = RandomizedSearchCV(catboost_clf, model_params, n_iter=10, cv=5, random_state=2489,verbose=1)

# train the random search meta-estimator to find the best model out of 100 candidates
model = clf.fit(tfidf_vectorizer_vectors,y)

# print winning set of hyperparameters
from pprint import pprint
pprint(model.best_estimator_.get_params())

print('Fullset Accuracy')
evaluation(model, tfidf_vectorizer_vectors,y)

print('Training Accuracy')
evaluation(model, train_x, train_y)
print('\nTesting Accuracy')
evaluation(model, valid_x, valid_y)

import joblib
# save the model to disk
filename = 'model_87_percent.pkl'
joblib.dump(model.best_estimator_, filename)

# some time later...
filename = 'model_87_percent.pkl'
# load the model from disk
loaded_model = joblib.load(filename)
print('\nTesting Accuracy')
evaluation(loaded_model, valid_x, valid_y)

pprint(model.best_estimator_.get_params())

predictions = loaded_model.predict(tfidf_vectorizer_vectors_Unseen)
len(predictions)

test_corpus['labels']=predictions
test_corpus.head()

# load the model from disk
loaded_model = joblib.load(filename)
print('\nTesting Accuracy')
evaluation(loaded_model, valid_x, valid_y)

predictions = loaded_model.predict(tfidf_vectorizer_vectors_Unseen)
len(predictions)

sample_submission = pd.read_csv('/content/sample_submission.csv',index_col=None)
# replacing the label with prediction
sample_submission['labels'] = predictions
sample_submission.head()

# saving the file
sample_submission.to_csv('catboost_87.csv', index=False)

sample_submission.head()

import pandas as pd

dft=pd.read_csv('/content/catboost_87.csv')

dft.head()
