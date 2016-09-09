from __future__ import print_function
import pprint
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import re
import sgmllib
import sys
reload(sys)
sys.setdefaultencoding('ISO-8859-1')


def _not_in_sphinx():

    return '__file__' in globals()

class ReutersParser(sgmllib.SGMLParser):
    """
    Used to get the topic(the first word) and boy of a sgm
    document one each time, and the input type is tuple in list
    [([topic1],body1),([topic2],body2)....]
    """
    def __init__(self, verbose=0):
        sgmllib.SGMLParser.__init__(self, verbose)
        self._reset()
    def _reset(self):#set the start parameter
        self.in_body = 0
        self.in_topics = 0
        self.in_topic_d = 0
        self.body = ""
        self.topics = []
        self.topic_d = ""
    def parse(self, fd):
        self.docs = []
        for chunk in fd:
            self.feed(chunk)
            for doc in self.docs:
                yield doc
            self.docs = []
        self.close()
    def handle_data(self, data):#get the data in body and topics
        if self.in_body:
            self.body += data
        elif self.in_topics:
            self.topic_d += data
    def start_reuters(self, attributes):
        pass
    def end_reuters(self):
        self.docs.append((self.topics, self.body))
        self._reset()

    def start_topics(self, attributes):
        self.in_topics = 1

    def end_topics(self):
        self.in_topics = 0

    def start_d(self, attributes):
        self.in_topic_d = 1

    def end_d(self):
        self.topics.append(self.topic_d)
        self.topic_d = ""

    def start_body(self, attributes):
        self.in_body = 1

    def end_body(self):
        self.in_body = 0





def tokenize(text):
    min_length = 3
    words = map(lambda word: word.lower(), word_tokenize(text));

    tokens =(list(map(lambda token: PorterStemmer().stem(token),
                  words)));
    p = re.compile('[a-zA-Z]+');
    filtered_tokens = filter(lambda token:
                  p.match(token) and len(token)>=min_length,
         tokens);
    return filtered_tokens

def obtain_topic_tags():
     categories = ['money','fx', 'crude','grain','trade', 'interest', 'wheat',
                   'ship', 'corn', 'oil', 'dlr', 'gas', 'oilseed', 'supply',
                   'sugar', 'gnp', 'coffee', 'veg', 'gold', 'soybean', 'bop',
'livestock', 'cpi','money-fx','money-supply','veg-oil']
     return categories
def filter_topics(categories,docs):
    ref_docs = []
    for d in docs:
        if d[0] == [] or d[0] == "" or d[1]=="":
            continue
        for n in d[0]:
            if n in categories:
                d_tup = (categories.index(n), d[1])
                ref_docs.append(d_tup)
                break
    return ref_docs

if __name__ == "__main__":
    filename = ["reuters21578/reut2-%03d.sgm" % r for r in range(0, 22)]
    #Set a loop to get all the document in 22 sgm files and print the
    #first document in the first file as an example to show the result
    #Also print the number of files to check whether if it is right to work
    parser = ReutersParser()
    docs = []
    for fn in filename:
        for d in parser.parse(open(fn,'rb')):
            docs.append(d)
    print (docs[1])
    print (len(docs))

    categories=obtain_topic_tags()
    docs_filter=filter_topics(categories,docs)
    print(docs_filter[0])
    print(len(docs_filter))


    topic_list = []
    body_list = []
    docs_token = []
    # #transfor the tokenize words into list
    length=len(docs_filter)
    for i in range(0,length):
        body = ''
        body_list= tokenize(docs_filter[i][1])
        for r in range(0,len(body_list)):
            body += body_list[r]+' '
        topic = str(docs_filter[i][0])
        docs_token.append(topic+','+body)
    print (docs_token[0])



    with open("trainingdata.txt",'w') as f:
        for s in docs_token:
            f.write(s + '\n')




