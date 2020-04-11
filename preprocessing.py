'''
function to perform preprocessing

'''

#Imports
import pandas as pd
import re
from stop_words import get_stop_words
from nltk.stem import WordNetLemmatizer


dataset_tag = ["QUESTION_ID_INDEX", "QUESTION_INDEX", "DOCUMENT_ID_INDEX", "DOCUMENT_TITLE_INDEX", "SENTENCE_ID_INDEX", "SENTENCE_INDEX", "LABEL_INDEX" ]   #Name Tags 

#load English Stopwords
stop_words = get_stop_words('en')
def remove_stopWords(s):
    '''For removing stop words
    '''
    s = ' '.join(word for word in s.split() if word not in stop_words)
    return s


#load wordnet lemmatizer
lemmatizer = WordNetLemmatizer()
def func_lemmatize(s):
    '''For lemmatizing
    '''
    s = ' '.join(lemmatizer.lemmatize(word) for word in s.split())
    return s

def preprocessing(dataset_rows, exclude_absolute_incorrect_question_groups = False):
    print("Entering Preprocessing Block")

    QUESTION_ID_INDEX = 0
    QUESTION_INDEX = 1
    DOCUMENT_ID_INDEX = 2
    DOCUMENT_TITLE_INDEX = 3
    SENTENCE_ID_INDEX = 4
    SENTENCE_INDEX = 5
    LABEL_INDEX = 6
    
    dataset_previous_tag = dataset_rows[0]  #store previous tags
    

    #Create Pandas dataframe for preprocessing
    dataframe = pd.DataFrame(dataset_rows[1:], columns = dataset_tag)
    #print(dataframe)

    dataframe = dataframe.drop(['DOCUMENT_ID_INDEX', 'SENTENCE_ID_INDEX'], axis = 1) #Remove non required columns

    #Convert Labels to integer
    dataframe["LABEL_INDEX"] = dataframe["LABEL_INDEX"].astype(int)
    #Convert all text data to lower
    dataframe.loc[:,"QUESTION_INDEX"] = dataframe.QUESTION_INDEX.apply(lambda x : str.lower(x))
    dataframe.loc[:,"SENTENCE_INDEX"] = dataframe.SENTENCE_INDEX.apply(lambda x : str.lower(x))

    #Remove punction
    dataframe.loc[:,"QUESTION_INDEX"] = dataframe.QUESTION_INDEX.apply(lambda x : " ".join(re.sub(r'[^\w]', ' ', x).split()))
    dataframe.loc[:,"SENTENCE_INDEX"] = dataframe.SENTENCE_INDEX.apply(lambda x : " ".join(re.sub(r'[^\w]', ' ', x).split()))

    #Remove Stopwords
    dataframe.loc[:,"QUESTION_INDEX"] = dataframe.QUESTION_INDEX.apply(lambda x : remove_stopWords(x))
    dataframe.loc[:,"SENTENCE_INDEX"] = dataframe.SENTENCE_INDEX.apply(lambda x : remove_stopWords(x))
    
    #Lemmatize
    dataframe.loc[:,"QUESTION_INDEX"] = dataframe.QUESTION_INDEX.apply(lambda x : func_lemmatize(x))
    dataframe.loc[:,"SENTENCE_INDEX"] = dataframe.SENTENCE_INDEX.apply(lambda x : func_lemmatize(x))

    dataframe_list = [dataframe.columns.values.tolist()] + dataframe.values.tolist()
    
    #Remove absolute incorrect question groups    
    if exclude_absolute_incorrect_question_groups == True:
        groups = dataframe.groupby("QUESTION_ID_INDEX").filter(lambda x: x["LABEL_INDEX"].sum() > 0)
        dataframe_list = [groups.columns.values.tolist()] + groups.values.tolist()
        print("Entered")

    return dataframe_list


#Remove absolute incorrect question groups
def drop_absolute_incorrect_question_groups(dataset_rows):
    dataframe = pd.DataFrame(dataset_rows)
    dataframe = dataframe.rename(columns=dataframe.iloc[0]).drop(dataframe.index[0])
    
    groups = dataframe.groupby("QUESTION_ID_INDEX").filter(lambda x: x["LABEL_INDEX"].sum() > 0)
    dataframe_list = groups.values.tolist()
    return dataframe_list
    

