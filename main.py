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
from wordcloud import WordCloud

#user defined module imports
import preprocessing
import jaccard_similarity
import cosine_similarity
import wordnet_similarity
import w2v_wordmoverdistance

#Indexing for Column headers 
QUESTION_ID_INDEX = 0
QUESTION_INDEX = 1
DOCUMENT_TITLE_INDEX = 2
SENTENCE_INDEX = 3
LABEL_INDEX = 4

#Dataset path
DATASET = "./dataset/zendesk_challenge.tsv"

#Initialization
dataset_rows = [] # list consisting of each row of the dataset
#list to store random positive labels
random_positive_labels_sensitivity_scores_list = []
random_positive_labels_precision_scores_list = []
random_positive_labels_f1_scores_list = []
random_positive_labels_exc_unrelated_sensitivity_scores_list = []
random_positive_labels_exc_unrelated_precision_scores_list = []
random_positive_labels_exc_unrelated_f1_scores_list = []

rand_number_count = 100 #Number of times random labels are to be generated before taking the highest value

#Dictionaries to store Sensitivity, Precision and F1 score computed on the entire dataset
dict_sensitivity_scores = {}
dict_precision_scores = {}
dict_f1_scores = {}

#Dictionaries to store Sensitivity, Precision and F1 score computed on the dataset with excludes question groups with no related answers
dict_exc_unrelated_sensitivity_scores = {}
dict_exc_unrelated_precision_scores = {}
dict_exc_unrelated_f1_scores = {}



def rand_bin_array(K, N):
    '''Function to generate list of random binary values
    K: number of 1's to be randomly generated
    N: Length of the list
    return: array of random variables
    '''
    arr = np.zeros(N)
    arr[:K]  = 1
    np.random.shuffle(arr)
    return arr.tolist()

#Function to plot bar plots using Dictionaries
def plot(dict_plot, name, filename):
    '''
    Function to plot the metric values
    dict_plot: Dictionaries of scores - type:dict
    name: Metric name - type:string
    filename: Filename for saving - type:string
    return: None
    '''
    m = 0
    map = []
    
    for i in sorted(dict_plot):
        print("{} = {}" .format(i, dict_plot[i]))
        plt.bar([m], dict_plot[i], color = 'C0')
        map.append(i)
        m += 1
    plt.xticks([r for r in range(len(map))], [key for key in map], rotation = -10, ha = 'left')
    plt.title(name)
    plt.ylabel('Score')
    plt.xlabel('Method')
    plt.savefig(filename)
    plt.show()

def generate_wordcloud_complete(data, index1, index2, filename):
    '''
    Function to plot Word CLoud
    data: text data - type: list
    index1: Index for text - type: int
    index2: Index for text 2- type: int
    filename: Filename for saving - type:string
    return: None
    '''
    text = []
    for row in data:
        text.append((row[index1] + " " + row[index2] + " ").split())
    
    text = [item for sublist in text for item in sublist]
    wordcloud = WordCloud().generate((" ").join(text))

    # Display the generated image:
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig(filename)
    plt.show()

def generate_wordcloud_partial(data, index,  filename):
    '''
    Function to plot Word CLoud
    data: text data - type: list
    index: Index for text - type: int
    filename: Filename for saving - type:string
    return: None
    '''
    text = []
    for row in data:
        text.append((row[index] + " ").split())
    
    text = [item for sublist in text for item in sublist]
    wordcloud = WordCloud().generate((" ").join(text))

    # Display the generated image:
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig(filename)
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
positive_label_count = sum(label) #number of positive labels to be present in the random label generator

for i in range(rand_number_count):
    #generate random positive labels
    random_positive_labels = rand_bin_array(positive_label_count, len(label))
    
    #Calculate Sensitivity Score and append to list
    random_positive_labels_sensitivity_scores = imblearn.metrics.sensitivity_score(label, random_positive_labels )
    random_positive_labels_sensitivity_scores_list.append(random_positive_labels_sensitivity_scores)
    
    #Calculate Precision Score and append to list
    random_positive_labels_precision_scores = sklearn.metrics.precision_score(label, random_positive_labels)
    random_positive_labels_precision_scores_list.append(random_positive_labels_precision_scores)
    
    #Calculate F1 Score and append to list
    random_positive_labels_f1_scores = sklearn.metrics.f1_score(label, random_positive_labels)
    random_positive_labels_f1_scores_list.append(random_positive_labels_f1_scores)

#Take max value of all the calculated random scores
dict_sensitivity_scores["Random Labels"] = max(random_positive_labels_sensitivity_scores_list)
dict_precision_scores["Random Labels"] = max(random_positive_labels_sensitivity_scores_list)
dict_f1_scores["Random Labels"] = max(random_positive_labels_sensitivity_scores_list)

#Calculate Jaccard Scores
dict_sensitivity_scores["Jaccard Similarity"] = imblearn.metrics.sensitivity_score(label, jaccard_prediction )
dict_precision_scores["Jaccard Similarity"] = sklearn.metrics.precision_score(label, jaccard_prediction)
dict_f1_scores["Jaccard Similarity"] = sklearn.metrics.f1_score(label, jaccard_prediction)


##########################################################################################################################

#Get prediction values from Jaccard Similarity excluding absolute incorrect question groups (groups with no single correct sentence)

label_exc_unrelated, jaccard_prediction_exc_unrelated = jaccard_similarity.model(dataset_absolute_incorrect_groups_dropped[1:])

#random-prediction statistics
positive_label_count_exc_unrelated = sum(label_exc_unrelated) #number of positive labels to be present in the random label generator

for i in range(rand_number_count):
    #generate random positive labels
    random_positive_labels_exc_unrelated = rand_bin_array(positive_label_count_exc_unrelated, len(label_exc_unrelated))
    
    #generate random positive labels
    random_positive_labels_exc_unrelated_sensitivity_scores = imblearn.metrics.sensitivity_score(label_exc_unrelated, random_positive_labels_exc_unrelated )
    random_positive_labels_exc_unrelated_sensitivity_scores_list.append(random_positive_labels_exc_unrelated_sensitivity_scores)
    
    #Calculate Precision Score and append to list
    random_positive_labels_exc_unrelated_precision_scores = sklearn.metrics.precision_score(label_exc_unrelated, random_positive_labels_exc_unrelated)
    random_positive_labels_exc_unrelated_precision_scores_list.append(random_positive_labels_exc_unrelated_precision_scores)
    
    #Calculate F1 Score and append to list
    random_positive_labels_exc_unrelated_f1_scores = sklearn.metrics.f1_score(label_exc_unrelated, random_positive_labels_exc_unrelated)
    random_positive_labels_exc_unrelated_f1_scores_list.append(random_positive_labels_exc_unrelated_f1_scores)

#Take max value of all the calculated random scores
dict_exc_unrelated_sensitivity_scores["Random Labels"] = max(random_positive_labels_exc_unrelated_sensitivity_scores_list)
dict_exc_unrelated_precision_scores["Random Labels"] = max(random_positive_labels_exc_unrelated_sensitivity_scores_list)
dict_exc_unrelated_f1_scores["Random Labels"] = max(random_positive_labels_exc_unrelated_sensitivity_scores_list)

#Calculate Jaccard Scores
dict_exc_unrelated_sensitivity_scores["Jaccard Similarity"] = imblearn.metrics.sensitivity_score(label_exc_unrelated, jaccard_prediction_exc_unrelated )
dict_exc_unrelated_precision_scores["Jaccard Similarity"] = sklearn.metrics.precision_score(label_exc_unrelated, jaccard_prediction_exc_unrelated)
dict_exc_unrelated_f1_scores["Jaccard Similarity"] = sklearn.metrics.f1_score(label_exc_unrelated, jaccard_prediction_exc_unrelated)




'''
Cosine Similarity
'''

#Cosine Similarity using CV
cosine_prediction_CV_complete_dataset, cosine_prediction_CV_2gram_complete_dataset = cosine_similarity.model(dataset[1:])
dict_sensitivity_scores["Cosine Similarity CV ngram-1"] = imblearn.metrics.sensitivity_score(label, cosine_prediction_CV_complete_dataset )
dict_precision_scores["Cosine Similarity CV ngram-1"] = sklearn.metrics.precision_score(label, cosine_prediction_CV_complete_dataset)
dict_f1_scores["Cosine Similarity CV ngram-1"] = sklearn.metrics.f1_score(label, cosine_prediction_CV_complete_dataset)

#Cosine Similarity using CV with n-gram = (1,2)
dict_sensitivity_scores["Cosine CV ngram-2"] = imblearn.metrics.sensitivity_score(label, cosine_prediction_CV_2gram_complete_dataset )
dict_precision_scores["Cosine CV ngram-2"] = sklearn.metrics.precision_score(label, cosine_prediction_CV_2gram_complete_dataset)
dict_f1_scores["Cosine CV ngram-2"] = sklearn.metrics.f1_score(label, cosine_prediction_CV_2gram_complete_dataset)


#Complete Unrelated Groups
#Cosine Similarity using CV
cosine_prediction_CV_exc_unrelated_dataset, cosine_prediction_CV_2gram_exc_unrelated_dataset  = cosine_similarity.model(dataset_absolute_incorrect_groups_dropped[1:])
dict_exc_unrelated_sensitivity_scores["Cosine CV ngram-1"] = imblearn.metrics.sensitivity_score(label_exc_unrelated, cosine_prediction_CV_exc_unrelated_dataset)
dict_exc_unrelated_precision_scores["Cosine CV ngram-1"] = sklearn.metrics.precision_score(label_exc_unrelated, cosine_prediction_CV_exc_unrelated_dataset)
dict_exc_unrelated_f1_scores["Cosine CV ngram-1"] = sklearn.metrics.f1_score(label_exc_unrelated, cosine_prediction_CV_exc_unrelated_dataset)

#Cosine Similarity using CV with n-gram = (1,2)
dict_exc_unrelated_sensitivity_scores["Cosine CV ngram-2"] = imblearn.metrics.sensitivity_score(label_exc_unrelated, cosine_prediction_CV_2gram_exc_unrelated_dataset )
dict_exc_unrelated_precision_scores["Cosine CV ngram-2"] = sklearn.metrics.precision_score(label_exc_unrelated, cosine_prediction_CV_2gram_exc_unrelated_dataset)
dict_exc_unrelated_f1_scores["Cosine CV ngram-2"] = sklearn.metrics.f1_score(label_exc_unrelated, cosine_prediction_CV_2gram_exc_unrelated_dataset)
#################################################################


'''
Wordnet Similarity
'''

#Wordnet Similarity
wordnet_similarity_complete_dataset = wordnet_similarity.model(dataset[1:])
dict_sensitivity_scores["Wordnet Similarity"] = imblearn.metrics.sensitivity_score(label, wordnet_similarity_complete_dataset )
dict_precision_scores["Wordnet Similarity"] = sklearn.metrics.precision_score(label, wordnet_similarity_complete_dataset)
dict_f1_scores["Wordnet Similarity"] = sklearn.metrics.f1_score(label, wordnet_similarity_complete_dataset)

# ####################################################################################

#Complete Unrelated Groups
#Wordnet Similarity
wordnet_similarity_exc_unrelated_dataset = wordnet_similarity.model(dataset_absolute_incorrect_groups_dropped[1:])
dict_exc_unrelated_sensitivity_scores["Wordnet Similarity"] = imblearn.metrics.sensitivity_score(label_exc_unrelated, wordnet_similarity_exc_unrelated_dataset )
dict_exc_unrelated_precision_scores["Wordnet Similarity"] = sklearn.metrics.precision_score(label_exc_unrelated, wordnet_similarity_exc_unrelated_dataset)
dict_exc_unrelated_f1_scores["Wordnet Similarity"] = sklearn.metrics.f1_score(label_exc_unrelated, wordnet_similarity_exc_unrelated_dataset)


'''
Word2Vec Word Mover distance
'''

#Using Common Bag of Words
w2v_word_mover_cbow_similarity, w2v_word_mover_sg_similarity  = w2v_wordmoverdistance.model(dataset[1:])
dict_sensitivity_scores["W2V WordMover CBOW"] = imblearn.metrics.sensitivity_score(label, w2v_word_mover_cbow_similarity )
dict_precision_scores["W2V WordMover CBOW"] = sklearn.metrics.precision_score(label, w2v_word_mover_cbow_similarity)
dict_f1_scores["W2V WordMover CBOW"] = sklearn.metrics.f1_score(label, w2v_word_mover_cbow_similarity)

#Using Skip Gram
dict_sensitivity_scores["W2V WordMover SG"] = imblearn.metrics.sensitivity_score(label, w2v_word_mover_cbow_similarity )
dict_precision_scores["W2V WordMover SG"] = sklearn.metrics.precision_score(label, w2v_word_mover_cbow_similarity)
dict_f1_scores["W2V WordMover SG"] = sklearn.metrics.f1_score(label, w2v_word_mover_cbow_similarity)

####################################################################

#Complete Unrelated Groups
#Using Common Bag of Words
w2v_word_mover_cbow_similarity_exc_unrelated_dataset, w2v_word_mover_sg_similarity_exc_unrelated_dataset  = w2v_wordmoverdistance.model(dataset_absolute_incorrect_groups_dropped[1:])
dict_exc_unrelated_sensitivity_scores["W2V WordMover CBOW"] = imblearn.metrics.sensitivity_score(label_exc_unrelated, w2v_word_mover_cbow_similarity_exc_unrelated_dataset )
dict_exc_unrelated_precision_scores["W2V WordMover CBOW"] = sklearn.metrics.precision_score(label_exc_unrelated, w2v_word_mover_cbow_similarity_exc_unrelated_dataset)
dict_exc_unrelated_f1_scores["W2V WordMover CBOW"] = sklearn.metrics.f1_score(label_exc_unrelated, w2v_word_mover_cbow_similarity_exc_unrelated_dataset)

#Using Skip Gram
dict_exc_unrelated_sensitivity_scores["W2V WordMover SG"] = imblearn.metrics.sensitivity_score(label_exc_unrelated, w2v_word_mover_cbow_similarity_exc_unrelated_dataset )
dict_exc_unrelated_precision_scores["W2V WordMover SG"] = sklearn.metrics.precision_score(label_exc_unrelated, w2v_word_mover_cbow_similarity_exc_unrelated_dataset)
dict_exc_unrelated_f1_scores["W2V WordMover SG"] = sklearn.metrics.f1_score(label_exc_unrelated, w2v_word_mover_cbow_similarity_exc_unrelated_dataset)


'''
Visualization
'''

#Plotting
plot(dict_sensitivity_scores, "Sensitivity Scores", "Complete_dataset_Sensitivity.png")
plot(dict_precision_scores, "Precision Scores", "Complete_dataset_Precision.png")
plot(dict_f1_scores, "F1 Scores", "Complete_dataset_F1.png")
plot(dict_exc_unrelated_sensitivity_scores, "Sensitivity Scores excluding complete irrelevant questions groups", "Exc_unrelated_Sensitivity.png")
plot(dict_exc_unrelated_precision_scores, "Precision Scores excluding complete irrelevant questions groups", "Exc_unrelated_Precision.png")
plot(dict_exc_unrelated_f1_scores, "F1 Scores excluding complete irrelevant questions groups", "Exc_unrelated_F1.png")

#Generate Word cloud
generate_wordcloud_complete(dataset[1:], QUESTION_INDEX, SENTENCE_INDEX, "WordCloud_all_text.png")
generate_wordcloud_partial(dataset[1:], QUESTION_INDEX, "WordCloud_Question.png")
generate_wordcloud_partial(dataset[1:], SENTENCE_INDEX, "WordCloud_Sentence.png")

#Print Scores
print(dict_sensitivity_scores)
print(dict_precision_scores)
print(dict_f1_scores)

print(dict_exc_unrelated_sensitivity_scores)
print(dict_exc_unrelated_precision_scores)
print(dict_exc_unrelated_f1_scores)