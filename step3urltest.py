# from __future__ import print_function
# import pprint
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import os
import re
import os.path
os.environ["SPARK_HOME"] = "/usr/local/spark"
os.environ["PYSPARK_PYTHON"]="/System/Library/Frameworks/Python.framework/Versions/2.7/bin/python2.7"
from pyspark import SparkContext
from pyspark.mllib.feature import HashingTF, IDF
from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
from pyspark.mllib.regression import LabeledPoint

import sys
reload(sys)
sys.setdefaultencoding('utf8')

cachedStopWords = stopwords.words("english")

def tokenize(text):
    min_length = 3
    words = map(lambda word: word.lower(), word_tokenize(text));
    words = [word for word in words
                  if word not in cachedStopWords]
    tokens =(list(map(lambda token: PorterStemmer().stem(token),
                  words)));
    p = re.compile('[a-zA-Z]+');
    filtered_tokens = filter(lambda token:
                  p.match(token) and len(token)>=min_length,
         tokens);
    return filtered_tokens

from goose import Goose

if __name__ == "__main__":
    url = 'http://www.reuters.com/article/global-oil-idUSL3N16408T'
    g = Goose()
    article = g.extract(url=url)
    a = article.cleaned_text
    html_dict = []
    tokenhtml = tokenize(a)
    print(tokenhtml)
    for i in range(0,len(tokenhtml)):
        body = ''
        body += tokenhtml[i]+' '
    html_dict.append({"label":"0","text":body})


    sc = SparkContext()
    htmldata = sc.parallelize(html_dict)
    labels = htmldata.map(lambda doc: doc["label"], preservesPartitioning = True)

    tf = HashingTF().transform(htmldata.map(lambda doc: doc["text"], preservesPartitioning=True))
    idf = IDF().fit(tf)
    tfidf = idf.transform(tf)
    end_tfidf = datetime.now()
    tfidf_time = format(end_tfidf-start_tfidf)

    dataset = labels.zip(tfidf).map(lambda x: LabeledPoint(x[0], x[1]))
    sameModel = NaiveBayesModel.load(sc, "/Users/apple/Dropbox/2016Spring/COSC526/MacHW1/mymodel")
    start_predict = datetime.now()
    predictionAndLabel = dataset.map(lambda p: (sameModel.predict(p.features)))  


    predict_time = format(end_predict-start_predict)
    accuracy = 1.0 * predictionAndLabel.filter(lambda (x, v): x == v).count() / dataset.count()


    print(tfidf_time)
   
    print(accuracy)