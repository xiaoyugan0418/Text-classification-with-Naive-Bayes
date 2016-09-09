from pyspark.mllib.feature import HashingTF, IDF
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import NaiveBayes
from pyspark import SparkContext
from datetime import datetime

def parseLine(line):
    dataset = []
    parts = line.split(',')
    labels = float(parts[0])
    features = parts[1]
    dataset.append({"label":labels,"text":features})
    return dataset

sc = SparkContext()
data = sc.textFile("trainingdata.txt").map(parseLine)


# Split data into labels and features, transform
# preservesPartitioning is not really required
# since map without partitioner shouldn't trigger repartitiong

labels = data.map(lambda doc: doc[0]["label"], preservesPartitioning = True)

for x in labels.take(3):
    print x
tf = HashingTF().transform(data.map(lambda doc: doc[0]["text"], preservesPartitioning=True))

idf = IDF().fit(tf)
tfidf = idf.transform(tf)

# Combine using zip
dataset = labels.zip(tfidf).map(lambda x: LabeledPoint(x[0], x[1]))

for x in dataset.take(3):
    print(x)
result=[]
start = datetime.now()
for number in range(0,10):
    training, test = dataset.randomSplit([0.6,0.4],seed=number)
    model = NaiveBayes.train(training,1.0)
    predictionAndLabel = test.map(lambda p: (model.predict(p.features), p.label))
    accuracy = 1.0 * predictionAndLabel.filter(lambda (x, v): x == v).count() / test.count()
    result.append(accuracy)
    print(accuracy)
print(result)
end = datetime.now()
time = end-start
print(time)
model.save(sc,"mynewmodel")