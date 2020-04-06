'''
Execution instructions:
python3 main.py
'''

import warnings
warnings.filterwarnings("ignore")

#Basic imports
import numpy as np
import csv
import sklearn
import imblearn
import matplotlib.pyplot as plt


#user defined module imports
import preprocessing
import jaccard_similarity
import cosine_similarity
import wordnet_similarity

#Indexing for Column headers 
QUESTION_ID_INDEX = 0
QUESTION_INDEX = 1
DOCUMENT_TITLE_INDEX = 2
SENTENCE_INDEX = 3
LABEL_INDEX = 4

#Dataset path
DATASET = "./dataset/zendesk_challenge_long.tsv"

#Initialization
dataset_rows = [] # list consisting of each row of the dataset
random_positive_labels_sensitivity_scores_list = []
random_positive_labels_precision_scores_list = []
random_positive_labels_f1_scores_list = []
random_positive_labels_exc_unrelated_sensitivity_scores_list = []
random_positive_labels_exc_unrelated_precision_scores_list = []
random_positive_labels_exc_unrelated_f1_scores_list = []

#Dictionaries to store Sensitivity, Precision and F1 score computed on the entire dataset
dict_sensitivity_scores = {
    "Random Labels Sensitivity Score": 0,
    "Jaccard Similarity Sensitivity Score": 0,
    }

dict_precision_scores = {
    "Random Labels Precision Score": 0,
    "Jaccard Similarity Precision Score": 0,
    }

dict_f1_scores = {
    "Random Labels F1 Score": 0,
    "Jaccard Similarity F1 Score": 0,
    }

#Dictionaries to store Sensitivity, Precision and F1 score computed on the dataset with excludes question groups with no related answers
dict_exc_unrelated_sensitivity_scores = {
    "Random Labels Exc Unrelated Groups Sensitivity Score": 0,
    "Jaccard Similarity Exc Unrelated Groups Sensitivity Score": 0,
    }

dict_exc_unrelated_precision_scores = {
    "Random Labels Exc Unrelated Groups Precision Score": 0,
    "Jaccard Similarity Exc Unrelated Groups Precision Score": 0,
    }

dict_exc_unrelated_f1_scores = {
    "Random Labels Exc Unrelated Groups F1 Score": 0,
    "Jaccard Similarity Exc Unrelated Groups F1 Score": 0,
    }

def rand_bin_array(K, N):
    '''Function to generate list of random binary values
    K: number of 1's to be randomly generated
    N: Length of the list
    '''
    arr = np.zeros(N)
    arr[:K]  = 1
    np.random.shuffle(arr)
    return arr.tolist()

#Function to plot bar plots using Dictionaries
def plot(dict_plot, name):
    plt.bar(dict_plot.keys(), dict_plot.values())
    plt.xticks([r for r in range(len(dict_plot.keys()))], [key for key, value in dict_plot.items()], rotation = -5)
    plt.title(name)
    plt.show()



with open(DATASET, encoding="utf8", errors='ignore') as dataset_open:
    dataset = csv.reader(dataset_open, delimiter='\t')
    for row in dataset:
        dataset_rows.append(row)


dataset = preprocessing.preprocessing(dataset_rows,exclude_absolute_incorrect_question_groups = False)
dataset_absolute_incorrect_groups_dropped = preprocessing.drop_absolute_incorrect_question_groups(dataset)

'''
Jaccard Similarity

'''

#Get prediction values from Jaccard Similarity using the entire dataset
label, jaccard_prediction = jaccard_similarity.model(dataset[1:])

#random-prediction statistics
positive_label_count = sum(label)

for i in range(100):
    random_positive_labels = rand_bin_array(positive_label_count, len(label))
    
    random_positive_labels_sensitivity_scores = imblearn.metrics.sensitivity_score(label, random_positive_labels )
    random_positive_labels_sensitivity_scores_list.append(random_positive_labels_sensitivity_scores)
    
    random_positive_labels_precision_scores = sklearn.metrics.precision_score(label, random_positive_labels)
    random_positive_labels_precision_scores_list.append(random_positive_labels_precision_scores)
    
    random_positive_labels_f1_scores = sklearn.metrics.f1_score(label, random_positive_labels)
    random_positive_labels_f1_scores_list.append(random_positive_labels_f1_scores)

dict_sensitivity_scores["Random Labels Sensitivity Score"] = max(random_positive_labels_sensitivity_scores_list)
dict_precision_scores["Random Labels Precision Score"] = max(random_positive_labels_sensitivity_scores_list)
dict_f1_scores["Random Labels F1 Score"] = max(random_positive_labels_sensitivity_scores_list)

dict_sensitivity_scores["Jaccard Similarity Sensitivity Score"] = imblearn.metrics.sensitivity_score(label, jaccard_prediction )
dict_precision_scores["Jaccard Similarity Precision Score"] = sklearn.metrics.precision_score(label, jaccard_prediction)
dict_f1_scores["Jaccard Similarity F1 Score"] = sklearn.metrics.f1_score(label, jaccard_prediction)


##########################################################################################################################

#Get prediction values from Jaccard Similarity excluding absolute incorrect question groups (groups with no single correct sentence)

label_exc_unrelated, jaccard_prediction_exc_unrelated = jaccard_similarity.model(dataset_absolute_incorrect_groups_dropped[1:])

#random-prediction statistics
positive_label_count_exc_unrelated = sum(label_exc_unrelated)

for i in range(100):
    random_positive_labels_exc_unrelated = rand_bin_array(positive_label_count_exc_unrelated, len(label_exc_unrelated))
    
    random_positive_labels_exc_unrelated_sensitivity_scores = imblearn.metrics.sensitivity_score(label_exc_unrelated, random_positive_labels_exc_unrelated )
    random_positive_labels_exc_unrelated_sensitivity_scores_list.append(random_positive_labels_exc_unrelated_sensitivity_scores)
    
    random_positive_labels_exc_unrelated_precision_scores = sklearn.metrics.precision_score(label_exc_unrelated, random_positive_labels_exc_unrelated)
    random_positive_labels_exc_unrelated_precision_scores_list.append(random_positive_labels_exc_unrelated_precision_scores)
    
    random_positive_labels_exc_unrelated_f1_scores = sklearn.metrics.f1_score(label_exc_unrelated, random_positive_labels_exc_unrelated)
    random_positive_labels_exc_unrelated_f1_scores_list.append(random_positive_labels_exc_unrelated_f1_scores)

dict_exc_unrelated_sensitivity_scores["Random Labels Exc Unrelated Groups Sensitivity Score"] = max(random_positive_labels_exc_unrelated_sensitivity_scores_list)
dict_exc_unrelated_precision_scores["Random Labels Exc Unrelated Groups Precision Score"] = max(random_positive_labels_exc_unrelated_sensitivity_scores_list)
dict_exc_unrelated_f1_scores["Random Labels Exc Unrelated Groups F1 Score"] = max(random_positive_labels_exc_unrelated_sensitivity_scores_list)

dict_exc_unrelated_sensitivity_scores["Jaccard Similarity Exc Unrelated Groups Sensitivity Score"] = imblearn.metrics.sensitivity_score(label_exc_unrelated, jaccard_prediction_exc_unrelated )
dict_exc_unrelated_precision_scores["Jaccard Similarity Exc Unrelated Groups Precision Score"] = sklearn.metrics.precision_score(label_exc_unrelated, jaccard_prediction_exc_unrelated)
dict_exc_unrelated_f1_scores["Jaccard Similarity Exc Unrelated Groups F1 Score"] = sklearn.metrics.f1_score(label_exc_unrelated, jaccard_prediction_exc_unrelated)




'''
Cosine Similarity
'''

#Cosine Similarity using TF-IDF
cosine_prediction_tf_idf_complete_dataset = cosine_similarity.model_tf_idf(dataset[1:])
dict_sensitivity_scores["Cosine Similarity TF-IDF Sensitivity Score"] = imblearn.metrics.sensitivity_score(label, cosine_prediction_tf_idf_complete_dataset )
dict_precision_scores["Cosine Similarity TF-IDF Precision Score"] = sklearn.metrics.precision_score(label, cosine_prediction_tf_idf_complete_dataset)
dict_f1_scores["Cosine Similarity TF-IDF F1 Score"] = sklearn.metrics.f1_score(label, cosine_prediction_tf_idf_complete_dataset)


cosine_prediction_tf_idf_exc_unrelated_dataset = cosine_similarity.model_tf_idf(dataset_absolute_incorrect_groups_dropped[1:])
dict_exc_unrelated_sensitivity_scores["Cosine Similarity TF-IDF Exc Unrelated Groups Sensitivity Score"] = imblearn.metrics.sensitivity_score(label_exc_unrelated, cosine_prediction_tf_idf_exc_unrelated_dataset )
dict_exc_unrelated_precision_scores["Cosine Similarity TF-IDF Exc Unrelated Groups Precision Score"] = sklearn.metrics.precision_score(label_exc_unrelated, cosine_prediction_tf_idf_exc_unrelated_dataset)
dict_exc_unrelated_f1_scores["Cosine Similarity TF-IDF Exc Unrelated Groups F1 Score"] = sklearn.metrics.f1_score(label_exc_unrelated, cosine_prediction_tf_idf_exc_unrelated_dataset)

#################################################################

#Cosine Similarity using Glove Model
'''
cosine_similarity_glove_complete_dataset = cosine_similarity.model_glove(dataset[1:])
dict_sensitivity_scores["Cosine Similarity Glove Sensitivity Score"] = imblearn.metrics.sensitivity_score(label, cosine_similarity_glove_complete_dataset )
dict_precision_scores["Cosine Similarity Glove Precision Score"] = sklearn.metrics.precision_score(label, cosine_similarity_glove_complete_dataset)
dict_f1_scores["Cosine Similarity Glove F1 Score"] = sklearn.metrics.f1_score(label, cosine_similarity_glove_complete_dataset)

'''

'''
Wordnet Similarity
'''
wordnet_similarity_glove_complete_dataset = wordnet_similarity.model(dataset[1:])
dict_sensitivity_scores["Wordnet Similarity Glove Sensitivity Score"] = imblearn.metrics.sensitivity_score(label, wordnet_similarity_glove_complete_dataset )
dict_precision_scores["Wordnet Similarity Glove Precision Score"] = sklearn.metrics.precision_score(label, wordnet_similarity_glove_complete_dataset)
dict_f1_scores["Wordnet Similarity Glove F1 Score"] = sklearn.metrics.f1_score(label, wordnet_similarity_glove_complete_dataset)


'''
Visualization
'''

print(dict_sensitivity_scores)
print(dict_precision_scores)
print(dict_f1_scores)

print(dict_exc_unrelated_sensitivity_scores)
print(dict_exc_unrelated_precision_scores)
print(dict_exc_unrelated_f1_scores)

plot(dict_sensitivity_scores, "Sensitivity Scores")
plot(dict_precision_scores, "Precision Scores")
plot(dict_f1_scores, "F1 Scores")
plot(dict_exc_unrelated_sensitivity_scores, "Sensitivity Scores excluding complete irrelevant questions groups")
plot(dict_exc_unrelated_precision_scores, "Precision Scores excluding complete irrelevant questions groups")
plot(dict_exc_unrelated_f1_scores, "F1 Scores excluding complete irrelevant questions groups")
