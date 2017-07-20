import pandas as pd
import numpy as np
import string
import gensim
import pickle
import time

# import xgboost as xgb
import re
from sklearn import preprocessing
from sklearn import cross_validation as cv
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer

from gensim import models, corpora

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import roc_auc_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn import linear_model as lm

start = time.clock()

tokenizer = RegexpTokenizer(r'\w+')
p_stemmer = PorterStemmer()
en_stop = set(stopwords.words('english'))

def isNaN(x):
    return x!=x

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print "Topic #%d:" % topic_idx
        print " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
    print


def text_process_basic(text):

    text = text.replace('\n', ' ') #removing line separators
    text = re.sub(r' +', r' ', text) #removing spurious spaces
    text = text.lower() #lower-case text
    tokens = tokenizer.tokenize(text) #breaking text into tokens
    tokens = [ch for ch in tokens if ch not in string.punctuation] #removing punctuations
    stopped_tokens = [i for i in tokens if not i in en_stop] #removing stop words
    text = [p_stemmer.stem(i) for i in stopped_tokens] #stemming words
    return text

def vectorize(train_text, n_topics, passes, save_model=True):

    dictionary = corpora.Dictionary(train_text)
    corpus_train = [dictionary.doc2bow(text) for text in train_text]

    ldamodel = gensim.models.ldamodel.LdaModel(corpus_train, num_topics=n_topics, id2word = dictionary, passes=passes)

    if save_model==True:
        dictionary.save('dictionary.dict')
        ldamodel.save('lda_model')

    for l in ldamodel.print_topics(num_topics=number_of_topics):
        print l

    train_topics = ldamodel[corpus_train]
    train_arr = gensim.matutils.corpus2dense(train_topics, n_topics)

    train_arr = train_arr.transpose()
    print train_arr
    # print dictionary.token2id

    feat_df = pd.DataFrame(train_arr)
    # print vectorizer.vocabulary_
    # print feat_df
    return feat_df

number_of_topics = 15
passes = 15

content = pd.read_csv('train.csv', index_col=0)
content = content.sample(frac=1).reset_index(drop=True)
print content.head()

content = content[isNaN(content['Text'])==False]
content['Text'] = content['Text'].apply(lambda x: x.decode('utf-8').encode('ascii', 'replace'))
content['Text'] = content['Text'].apply(lambda x: text_process_basic(x))

topic_model_df = vectorize(content['Text'].tolist(), number_of_topics, passes, save_model=True)
dat = pd.concat([content, topic_model_df], axis=1)

step_2 = time.clock()
print step_2 - start
print
print dat.head()

dat['Label'] = dat['Label'].apply(lambda x: 1 if x=='Books' else 0)
print dat['Label'].value_counts()

# print dat.head()
dat.fillna(0, inplace=True)
dat_var = dat.drop(['Text', 'Label'], axis=1, inplace=False)
dat_label = dat['Label']

X_train, X_test, y_train, y_test = train_test_split(dat_var, dat_label, test_size=0.3, random_state=1)
# print X_train.head()

skf = cv.StratifiedKFold(y_train, n_folds=3, shuffle=True)
score_metric = 'roc_auc'
scores = {}
#
#
def score_model(model):
    return cv.cross_val_score(model, X_train, y_train, cv=skf, scoring=score_metric)
# #
# # Training Classifiers
# #
scores["Logistic"] = score_model(LogisticRegression())
print 'Logistic Done'
scores["RF"] = score_model(RandomForestClassifier())
print "RF done"
scores["Grad_Boost"] = score_model(GradientBoostingClassifier(n_estimators=100))
print "Grad Boost Done"
scores["Adaboost"] = score_model(AdaBoostClassifier(n_estimators=100))
print "Ada Boost Done"
# #
# #
model_scores = pd.DataFrame(scores).mean()
print model_scores

step_3 = time.clock()
print step_3 - step_2

def parameter_search(model, score='roc_auc'):

    if model=='LogisticRegression':
        tuned_parameters = [{'C': [1, 10, 100]}]

        print "# Tuning hyper-parameters for {}".format(score)

        clf = GridSearchCV(LogisticRegression(C=1), tuned_parameters, cv=5,
                           scoring='%s' % score)

    elif model=='GradientBoostingClassifier':

        tuned_parameters = [{'n_estimators': [150, 250, 500],'min_samples_split': [2, 30, 100]}]

        print "# Tuning hyper-parameters for {}".format(score)

        clf = GridSearchCV(GradientBoostingClassifier(n_estimators=100), tuned_parameters, cv=5,
                           scoring='%s' % score)

    clf.fit(X_train, y_train)

    print "Best parameters set found on development set:"

    print clf.best_params_

    print "Grid scores on development set:"
    print
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print "{} (+/-{}) for {}".format(mean, std * 2, params)

    y_true, y_pred = y_test, clf.predict(X_test)
    print roc_auc_score(y_true, y_pred)
    print

parameter_search('LogisticRegression', 'roc_auc')

parameter_search('GradientBoostingClassifier', 'roc_auc')

step_4 = time.clock()
print step_4 - step_3
# gb = GradientBoostingClassifier()
# gb = gb.fit(train_vars, train_class)
#
# pred = gb.predict_proba(test_vars)[:, 1]
Preds = {}

# clf_xgb = xgb.XGBClassifier(missing=np.nan, max_depth=3, min_child_weight=1, gamma=0, n_estimators=1400, learning_rate=0.05, nthread=4, subsample=0.90, colsample_bytree=0.85, scale_pos_weight=1, seed=4242)

# X_train, X_test, y_train, y_test = train_test_split(train_vars, train_class, test_size=0.28, random_state=42)

# clf_xgb.fit(X_train, y_train, early_stopping_rounds=50, eval_metric="auc", eval_set=[(X_test, y_test)])

clf_0 = LogisticRegression()
clf_1 = RandomForestClassifier(n_estimators=250)
clf_2 = GradientBoostingClassifier(n_estimators=250)
clf_3 = AdaBoostClassifier(n_estimators=250)


def predictions(xtrain, ytrain, xtest):
    clf = clf_0.fit(xtrain, ytrain)
    with open('finalized_model.sav', 'wb') as f:
        pickle.dump(clf_0, f)
    Preds["Logistic"] = clf.predict_proba(xtest)[:, 1]
    clf = clf_1.fit(xtrain, ytrain)
    Preds["RF"] = clf.predict_proba(xtest)[:, 1]
    clf = clf_2.fit(xtrain, ytrain)
    Preds["G_Boost"] = clf.predict_proba(xtest)[:, 1]
    clf = clf_3.fit(xtrain, ytrain)
    Preds["Ada_Boost"] = clf.predict_proba(xtest)[:, 1]
    # Preds["XG_Boost"] = clf_xgb.predict_proba(xtest)[:, 1]
    return pd.DataFrame(Preds)

Preds_1 = predictions(X_train, y_train, X_test)
print Preds_1.head()

print "Logistic = ", roc_auc_score(y_test, Preds_1["Logistic"])
print "RF = ", roc_auc_score(y_test, Preds_1["RF"])
print "G_Boost AUC = ", roc_auc_score(y_test, Preds_1["G_Boost"])
print "Ada_Boost = ", roc_auc_score(y_test, Preds_1["Ada_Boost"])

#Creating an Ensemble Model
combi_mod = lm.LogisticRegression(C=1e11)
combi_mod = combi_mod.fit(Preds_1, y_test)
#
pred = combi_mod.predict_proba(Preds_1)[:, 1]
#
print "Combi_Model AUC = ", roc_auc_score(y_test, pred)

pred_binary = combi_mod.predict(Preds_1)
print pred_binary[:5]

print metrics.accuracy_score(y_test, pred_binary)
print metrics.confusion_matrix(y_test, pred_binary)

step_5 = time.clock()
print step_5 - step_4
