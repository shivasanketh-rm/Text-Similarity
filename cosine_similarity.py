
from sklearn.feature_extraction.text import CountVectorizer
import scipy
from scipy.spatial import distance
import numpy as np
from nltk.corpus import stopwords
import re


QUESTION_ID_INDEX = 0
QUESTION_INDEX = 1
DOCUMENT_TITLE_INDEX = 2
SENTENCE_INDEX = 3
LABEL_INDEX = 4



def cosine_distance_countvectorizer_method(s1, s2, ngram):
    
    # list conversion
    agg_sentences = [s1 , s2]
    if s1 == "" or s2 == "":
        return 0
    
    # text to vector
    vectorizer = CountVectorizer(ngram_range=ngram)
    all_sentences_to_vector = vectorizer.fit_transform(agg_sentences)
    text_to_vector_v1 = all_sentences_to_vector.toarray()[0].tolist()
    text_to_vector_v2 = all_sentences_to_vector.toarray()[1].tolist()
    
    # distance of similarity
    cosine = distance.cosine(text_to_vector_v1, text_to_vector_v2)
    
    return (int(round(1-cosine)))



def model(data):
    print("Entering Cosine Similarity module")
    prediction_label_ngram1 = []
    prediction_label_ngram2 = []
    for row in data:
        prediction_label_ngram1.append(cosine_distance_countvectorizer_method(row[QUESTION_INDEX], row[SENTENCE_INDEX], (1, 1)))
        prediction_label_ngram2.append(cosine_distance_countvectorizer_method(row[QUESTION_INDEX], row[SENTENCE_INDEX], (1, 2)))

    return prediction_label_ngram1, prediction_label_ngram2



if __name__ == "__main__":
    ss1 = 'The president greets the press in Chicago'
    ss2 = 'Obama speaks to the media in Illinois and not'
    #cosine_distance_countvectorizer_method(ss1 , ss2)

    