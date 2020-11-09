from flask import Flask,render_template,url_for,request,Response
import pandas as pd 
import numpy as np
from pandas import DataFrame
from nltk.stem.porter import PorterStemmer
from nltk.corpus import movie_reviews
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy as nltk_accuracy
from nltk.corpus import stopwords
# from nltk.stem.wordnet import WordNetLemmatizer
import re
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

print('started')

## Definitions
def remove_pattern(input_txt,pattern):
    r = re.findall(pattern,input_txt)
    for i in r:
        input_txt = re.sub(i,'',input_txt)
    return input_txt
    
def count_punct(text):
    count = sum([1 for char in text if char in string.punctuation])
    return round(count/(len(text) - text.count(" ")),3)*100


def extract_features(words):
    return dict([(word, True) for word in words])


app = Flask(__name__)


data = pd.read_csv("sentiment.tsv",sep = '\t')
frame = DataFrame(data)
data.columns = ["label","body_text"]
# Features and Labels
data['label'] = data['label'].map({'pos': 0, 'neg': 1})
data['tidy_tweet'] = np.vectorize(remove_pattern)(data['body_text'],"@[\w]*")
tokenized_tweet = data['tidy_tweet'].apply(lambda x: x.split())
stemmer = PorterStemmer()
# Stemming - connection, connected, connecting word reduce to a common word "connect".
tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) 
for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])
data['tidy_tweet'] = tokenized_tweet
data['body_len'] = data['body_text'].apply(lambda x:len(x) - x.count(" "))
data['punct%'] = data['body_text'].apply(lambda x:count_punct(x))
X = data['tidy_tweet']
y = data['label']
print(type(X))
# Extract Feature With CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(X) # Fit the Data
X = pd.concat([data['body_len'],data['punct%'],pd.DataFrame(X.toarray())],axis = 1)
from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
## Using Classifier
clf = LogisticRegression(C=0.1, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, max_iter=100,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=None, solver='liblinear', tol=0.0001, verbose=0,
                   warm_start=False)
clf.fit(X,y)

def SentimentAnalyzer(text):
    # load movie reviews from sample data
    # fileids_pos = movie_reviews.fileids('pos')
    # fileids_neg = movie_reviews.fileids('neg')

    # features_pos = [(extract_features(movie_reviews.words(fileids=[f])),'Positive') for f in fileids_pos]
    # features_neg = [(extract_features(movie_reviews.words(fileids=[f])),'Negative') for f in fileids_neg]

    threshold = 0.8
    # num_pos = int(threshold*len(features_pos))
    # num_neg = int(threshold*len(features_neg))

    # creating training and testing data
    # features_train = features_pos[:num_pos] + features_neg[:num_neg]
    # features_test = features_pos[num_pos:] + features_neg[num_neg:]
    feature = frame.body_text
    label = frame.label
    features_train = [(extract_features(feature), label) for index, (feature, label) in frame.iterrows()]

    features_train = features_train[:2000] 
    features_test = features_train[2000:]

    print('\nNumber of training datapoints:', len(features_train))
    print('Number of test datapoints:', len(features_test))

    # training a naive bayes classifier 
    print(type(features_train))
    print(type(features_train[0]))
    print(type(features_train[0][0]))
    classifier = NaiveBayesClassifier.train(features_train)
    print('Accuracy:',nltk_accuracy(classifier, features_test))

    probabilities = classifier.prob_classify(extract_features(text.split()))
    # Pick the maximum value
    predicted_sentiment = probabilities.max()
    print("Predicted sentiment:", predicted_sentiment)
    print("Probability:",round(probabilities.prob(predicted_sentiment), 2))

    return predicted_sentiment 


@app.route('/',methods=['GET'])
def home():
    print('entering')
    SentimentAnalyzer('It was not that good.')
    return render_template('try.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = pd.DataFrame(cv.transform(data).toarray())
        body_len = pd.DataFrame([len(data) - data.count(" ")])
        punct = pd.DataFrame([count_punct(data)])
        total_data = pd.concat([body_len,punct,vect],axis = 1)
        my_prediction = clf.predict(total_data)
        # ai_predict = result(message)
        print(my_prediction)
        # print(ai_predict)
        d = {'my_prediction':my_prediction,'ai_predict':ai_predict}
    return render_template('result.html',prediction = d)


if __name__ == '__main__':
    app.run()
