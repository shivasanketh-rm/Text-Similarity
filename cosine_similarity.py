
from sklearn.feature_extraction.text import CountVectorizer
import scipy
from scipy.spatial import distance
import numpy as np
from nltk.corpus import stopwords
import re

gloveFile = "./model/Glove/glove.6B.50d.txt"


QUESTION_ID_INDEX = 0
QUESTION_INDEX = 1
DOCUMENT_TITLE_INDEX = 2
SENTENCE_INDEX = 3
LABEL_INDEX = 4

def loadGloveModel(gloveFile):
    print ("Loading Glove Model")
    with open(gloveFile, encoding="utf8" ) as f:
        content = f.readlines()
    model = {}
    for line in content:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print ("Done.",len(model)," words loaded!")
    return model

model = loadGloveModel(gloveFile)

def cosine_distance_countvectorizer_method(s1, s2):
    
    # list conversion
    agg_sentences = [s1 , s2]
    if s1 == "" or s2 == "":
        return 0
    
    # text to vector
    vectorizer = CountVectorizer()
    all_sentences_to_vector = vectorizer.fit_transform(agg_sentences)
    text_to_vector_v1 = all_sentences_to_vector.toarray()[0].tolist()
    text_to_vector_v2 = all_sentences_to_vector.toarray()[1].tolist()
    
    # distance of similarity
    cosine = distance.cosine(text_to_vector_v1, text_to_vector_v2)
    
    return (int(round(1-cosine)))

def preprocess(raw_text):

    # keep only words
    letters_only_text = re.sub("[^a-zA-Z]", " ", raw_text)

    # convert to lower case and split 
    words = letters_only_text.lower().split()

    # remove stopwords
    stopword_set = set(stopwords.words("english"))
    cleaned_words = list(set([w for w in words if w not in stopword_set]))
    print(cleaned_words)
    return cleaned_words

def pre_process(text):
    cleaned_words = text.split()
    return cleaned_words


def cosine_distance_wordembedding_method(s1, s2):
    if s1 == "" or s2 == "":
        return 0
    vector_1 = np.mean([model[word] for word in pre_process(s1)],axis=0)
    vector_2 = np.mean([model[word] for word in pre_process(s2)],axis=0)
    cosine = scipy.spatial.distance.cosine(vector_1, vector_2)
    #print('Word Embedding method with a cosine distance asses that our two sentences are similar to',round((1-cosine)*100,2),'%')
    return (int(round(1-cosine)))

def model_tf_idf(data):
    prediction_label = []
    for row in data:
        prediction_label.append(cosine_distance_countvectorizer_method(row[QUESTION_INDEX], row[SENTENCE_INDEX]))

    return prediction_label

def model_glove(data):
    prediction_label = []

    for row in data:
        prediction_label.append(cosine_distance_wordembedding_method(row[QUESTION_INDEX], row[SENTENCE_INDEX]))

    return prediction_label


if __name__ == "__main__":
    ss1 = 'The president greets the press in Chicago'
    ss2 = 'Obama speaks to the media in Illinois and not the ai of the world gog og'
    #cosine_distance_countvectorizer_method(ss1 , ss2)

    model = loadGloveModel(gloveFile)
    cosine_distance_wordembedding_method(ss1, ss2)