import warnings
import logging
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    for idx in range(len(test_set.get_all_Xlengths())):
        test_X, test_length = test_set.get_item_Xlengths(idx)
        test_word_probabilities = {}
        best_guess = ""
        best_score = float("-Inf")
        for word, model in models.items():
            try:
                score = model.score(test_X, test_length)
                test_word_probabilities[word] = score
                if score > best_score:
                    best_guess, best_score = word, score
            except Exception as e:
                #logging.exception('recognizer exception: ', e)
                test_word_probabilities[word] = float("-Inf")

        probabilities.append(test_word_probabilities)
        guesses.append(best_guess)


    return probabilities, guesses
