from nltk.tokenize import sent_tokenize, word_tokenize 
import warnings 

warnings.filterwarnings(action = 'ignore') 
  
import gensim 
from gensim.models import Word2Vec

QUESTION_ID_INDEX = 0
QUESTION_INDEX = 1
DOCUMENT_TITLE_INDEX = 2
SENTENCE_INDEX = 3
LABEL_INDEX = 4


def word_mover(text_model, s1, s2):
    word_mover_sim = text_model.wmdistance(s1, s2)
    if word_mover_sim == float("inf"):
        return 0
    return int(round(((int(round(word_mover_sim)) - 0) * (1 - 0)) / (5 - 0)))

def model(data):
    '''
    To find Word Mover Distance using Word2Vec
    data: Preprocessed dataset - list
    return:
    prediction_label_cbow: Word Mover Distance Prediction label with CBOW - list
    prediction_label_sg: Word Mover Distance Prediction label with Skip Gram - list
    '''

    print("Entering Word2Vec with Word Mover Distance module")
    prediction_label_cbow = []
    prediction_label_sg = []
    
    text = []
    for row in data:
        text.append((row[QUESTION_INDEX] + " " + row[SENTENCE_INDEX] + " ").split())
    
    text = [item for sublist in text for item in sublist]
    
    model_cbow = gensim.models.Word2Vec(text, min_count = 1, size = 100, window = 5) 
    model_sg = gensim.models.Word2Vec(text, min_count = 1, size = 100, window = 5, sg = 1) 
 
    
    for row in data:
        prediction_label_cbow.append(word_mover(model_cbow, row[QUESTION_INDEX], row[SENTENCE_INDEX]))
        prediction_label_sg.append(word_mover(model_sg, row[QUESTION_INDEX], row[SENTENCE_INDEX]))

    return prediction_label_cbow, prediction_label_sg