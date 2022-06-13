from django.shortcuts import render
from .apps import ApiConfig
from rest_framework.views import APIView
from rest_framework.response import Response
import pickle5 as pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.metrics import accuracy_score

class HoaxPredict(APIView):
    clf = pickle.load(open('pac_model.pkl', 'rb'))
    tf1 = pickle.load(open("tfidf1.pkl", 'rb'))

    X_test = pd.read_csv('testdata.csv', error_bad_lines=False, encoding='latin1')
    inputdata = input("Enter your text: ")


    # Create new tfidfVectorizer with old vocabulary
    factory = StopWordRemoverFactory()
    stopwords = factory.get_stop_words()
    tf1_new = TfidfVectorizer(analyzer='word', ngram_range=(1,2), stop_words=frozenset(stopwords), lowercase = True,
                          max_features = 500000, vocabulary = tf1.vocabulary_)
    X_tf1 = tf1_new.fit_transform([inputdata])

    pred = clf.predict(X_tf1)
    response_dict = {"predict of fact/hoax": pred}
    return Response(response_dict, status=200)


# Create your views here.
