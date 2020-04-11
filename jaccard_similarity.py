'''
function to find jaccard similarity

'''

QUESTION_ID_INDEX = 0
QUESTION_INDEX = 1
DOCUMENT_TITLE_INDEX = 2
SENTENCE_INDEX = 3
LABEL_INDEX = 4



def get_jaccard_sim(str1, str2): 
    '''
    To find Jaccard similarity
    str1: Sentence 1
    str2: Sentence 2
    return: Jaccard Similarity Score - Int
    '''
    a = set(str1.split()) 
    b = set(str2.split())
    c = a.intersection(b)
    score = len(c) / (len(a) + len(b) - len(c))
    if score > 0.5:
        return 1
    return 0
    #return len(c)) / (len(a) + len(b) - len(c)



def model(data):
    '''
    To find Jaccard Similarity and actual labels
    data: Preprocessed dataset - list
    return:
    actual_label: Actual label of dataset - list
    prediction_label: Jaccard Prediction label - list
    '''

    print("Entering Jaccard Similarity module")
    actual_label = []
    prediction_label = []
    for row in data:
        actual_label.append(int(row[LABEL_INDEX]))
        prediction_label.append(get_jaccard_sim(row[QUESTION_INDEX], row[SENTENCE_INDEX]))

    return actual_label, prediction_label