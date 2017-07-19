import pandas as pd
import numpy as np
import string
import gensim
import pickle
# import xgboost as xgb
import re
from sklearn import preprocessing
from sklearn import cross_validation as cv
from sklearn import metrics

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer

from gensim import models, corpora

tokenizer = RegexpTokenizer(r'\w+')
p_stemmer = PorterStemmer()
en_stop = set(stopwords.words('english'))

def isNaN(x):
    return x!=x

def text_process_basic(text):

    text = text.replace('\n', ' ')
    text = re.sub(r' +', r' ', text)
    text = text.lower()
    tokens = tokenizer.tokenize(text)
    tokens = [ch for ch in tokens if ch not in string.punctuation]
    stopped_tokens = [i for i in tokens if not i in en_stop]
    text = [p_stemmer.stem(i) for i in stopped_tokens]
    return text

def vectorize(test_text, n_topics, len = 1):

    dictionary = gensim.corpora.Dictionary.load('dictionary.dict')
    ldamodel = gensim.models.wrappers.LdaMallet.load('lda_model')

    #make topic printing work
    # print ldamodel.show_topics()

    if len==1:
        corpus_test = dictionary.doc2bow(test_text)
        test_topics = ldamodel[corpus_test]
        test_arr = [t[1] for t in test_topics]
        feat_df = np.reshape(test_arr, (1,-1))

    else:
        corpus_test = [dictionary.doc2bow(text) for text in test_text]
        test_topics = ldamodel[corpus_test]
        test_arr = gensim.matutils.corpus2dense(test_topics, n_topics)
        test_arr = test_arr.transpose()
        feat_df = pd.DataFrame(test_arr)

    return feat_df

def classify_text(model, feat_df):
    label = loaded_model.predict(feat_df)
    if label==1:
        return 'Book'
    else:
        return 'Movie'



do = raw_input('Write 0 for text input, 1 for csv input\n')
n_topics = 15

if do=='0':
    text = raw_input('Write your text below\n')
    text = text_process_basic(text)

    feats = vectorize(text, n_topics, len = 1)
    filename = 'finalized_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    label = loaded_model.predict(feat)
    print label

elif do=='1':
    f_path = raw_input('Give file path\n')

    content = pd.read_csv(f_path, index_col=0)
    len_ = len(content)
    content = content[isNaN(content['Text'])==False]
    content['Text'] = content['Text'].apply(lambda x: x.decode('utf-8').encode('ascii', 'replace'))
    content['Text'] = content['Text'].apply(lambda x: text_process_basic(x))

    topic_model_df = vectorize(content['Text'].tolist(), n_topics, len=len_)
    print topic_model_df.head()
    dat = pd.concat([content, topic_model_df], axis=1)
    print dat.head()

    dat['Label'] = dat['Label'].apply(lambda x: 1 if x=='Books' else 0)

    dat.fillna(0, inplace=True)
    test_var = dat.drop(['Text', 'Label'], axis=1, inplace=False)
    test_label = dat['Label']

    filename = 'finalized_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    pred_labels = loaded_model.predict(test_var)

    print 'roc_auc_score'
    print metrics.roc_auc_score(test_label, loaded_model.predict_proba(test_var)[:,1])
    print 'classification accuracy'
    print metrics.accuracy_score(test_label, pred_labels)
    print metrics.confusion_matrix(test_label, pred_labels)
