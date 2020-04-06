'''
function to find jaccard similarity

'''

#Imports


QUESTION_ID_INDEX = 0
QUESTION_INDEX = 1
DOCUMENT_TITLE_INDEX = 2
SENTENCE_INDEX = 3
LABEL_INDEX = 4



def get_jaccard_sim(str1, str2): 
    a = set(str1.split()) 
    b = set(str2.split())
    c = a.intersection(b)
    score = len(c) / (len(a) + len(b) - len(c))
    if score > 0.5:
        return 1
    return 0
    #return len(c)) / (len(a) + len(b) - len(c)



def model(data):
    actual_label = []
    prediction_label = []
    for row in data:
        actual_label.append(int(row[LABEL_INDEX]))
        prediction_label.append(get_jaccard_sim(row[QUESTION_INDEX], row[SENTENCE_INDEX]))

    return actual_label, prediction_label