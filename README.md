we use the popular Reuters 21578 collection of documents as our training dataset which 
can obtain the dataset from this site: http://archive.ics.uci.edu/ml/machine-learning-databases/reuters21578-mld/. 
The collection has a total of 123 categories. We modeling Naïve Bayes Classifier according to the following 
steps: 
1.Parse XML documents to extract topics and related content.
2. Tokenize the documents and stem.
3. Create our dictionary of all words (i.e., vocabulary) in 
the collection and obtain a inverse document frequency (IDF) for each term.
4. Vectorize documents using the TF-IDF scores
5. Train the NB classifier.
4. Classify HTML documents.
