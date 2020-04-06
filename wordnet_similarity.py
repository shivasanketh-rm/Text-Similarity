from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn


QUESTION_ID_INDEX = 0
QUESTION_INDEX = 1
DOCUMENT_TITLE_INDEX = 2
SENTENCE_INDEX = 3
LABEL_INDEX = 4

def penn_to_wn(tag):
    """ Convert between a Penn Treebank tag to a simplified Wordnet tag """
    if tag.startswith('N'):
        return 'n'
 
    if tag.startswith('V'):
        return 'v'
 
    if tag.startswith('J'):
        return 'a'
 
    if tag.startswith('R'):
        return 'r'
 
    return None
 
def tagged_to_synset(word, tag):
    wn_tag = penn_to_wn(tag)
    if wn_tag is None:
        return None
    try:
        return wn.synsets(word, wn_tag)[0]
    except:
        return None
 
def sentence_similarity(sentence1, sentence2):
    """ compute the sentence similarity using Wordnet """
    # Tokenize and tag
    sentence1 = pos_tag(word_tokenize(sentence1))
    sentence2 = pos_tag(word_tokenize(sentence2))
 
    # Get the synsets for the tagged words
    synsets1 = [tagged_to_synset(*tagged_word) for tagged_word in sentence1]
    synsets2 = [tagged_to_synset(*tagged_word) for tagged_word in sentence2]
 
    # Filter out the Nones
    synsets1 = [ss for ss in synsets1 if ss]
    synsets2 = [ss for ss in synsets2 if ss]
 
    score, count = 0.0, 0
 
    # For each word in the first sentence
    for synset in synsets1:
        # Get the similarity value of the most similar word in the other sentence
        best_score = [synset.path_similarity(ss) for ss in synsets2]
        best_score = [s for s in best_score if isinstance(s, float)]
        if best_score != []:
            best_score = max(best_score)
        else:
            best_score = 0
 
        # Check that the similarity could have been computed
        if best_score is not None:
            score += best_score
            count += 1
 
    # Average the values
    if count != 0:
        score /= count
    else:
        score = 0
    return (int(round(score)))
 
def model(data):
    prediction_label = []
    for row in data:
        prediction_label.append(sentence_similarity(row[QUESTION_INDEX], row[SENTENCE_INDEX]))
    return prediction_label

if __name__ == "__main__":
    ss1 = 'The president greets the press in Chicago'
    ss2 = 'Obama speaks to the media in Illinois and not the ai of the world gog og'
    print(sentence_similarity(ss1, ss2))
