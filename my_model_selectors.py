import math
import statistics
import warnings
import heapq
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences

import logging

class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        """ nParam: nn + 2n*d-1 
         where n = number of states
         d = number of data points
         from https://discussions.udacity.com/t/bayesian-information-criteria-equation/326887/3
        """
        calculateNParams = lambda n, d: n ** 2 + 2 * n * d - 1
        calculateBIC = lambda L, p, d: -2 * L + p * np.log(d)
        best_score = (float("Inf"), None)
        for n in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = self.base_model(n)
                L = model.score(self.X, self.lengths)
                d = model.n_features
                p = calculateNParams(n, d)
                score = calculateBIC(L, p, n)
                if score < best_score[0]:
                    best_score = (score, model)
            except:
                # catch the exception for the illegal transitions
                pass
        if not best_score[1]:
            return self.base_model(self.n_constant)
        return best_score[1]

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''
    def sum_scores(self, model, all_but):
        scores = [model.score(X, L) for X, L in all_but]
        return sum(scores)

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        
        all_but = [self.hwords[word] for word in self.words if word != self.this_word]
        base_scores = []
        for n in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = self.base_model(n)
                score = model.score(self.X, self.lengths)
                base_scores.append(tuple([score, model]))
            except Exception as e:
                #logging.exception('DIC exception occurred:', e)
                pass

        best_score = float("-Inf")
        best_model = None
        
        for score_model in base_scores:
            try:
                orig_score, orig_model = score_model[0], score_model[1]
                score = orig_score - np.mean(self.sum_scores(orig_model, all_but))     
                if score > best_score:
                    best_score, best_model = score, orig_model

            except Exception as e:
                # logging.exception('DIC exception occurred:', e)
                pass
        if not best_model:
            return self.base_model(self.n_constant)
        return best_model

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
                
        kf = KFold(n_splits = 3, shuffle = False, random_state = None)
        likelihoods = []
        best_score = float("-Inf")
        best_model = None

        for n_states in range(self.min_n_components, self.max_n_components + 1):
            try:        
                # Check sufficient data to split using KFold
                if len(self.sequences) < 3:
                    model = self.base_model(n_states)
                    likelihoods.append(model.score(self.X, self.lengths))
                else:
                    # loop through the splits and keep track of the likelihoods
                    for train_idx, test_idx in kf.split(self.sequences):
                        # Training sequences split using KFold are recombined
                        self.X, self.lengths = combine_sequences(train_idx, self.sequences)

                        # Test sequences split using KFold are recombined
                        X_test, lengths_test = combine_sequences(test_idx, self.sequences)

                        model = self.base_model(n_states)
                        likelihood = model.score(X_test, lengths_test)
                        likelihoods.append(likelihood)

                # get the mean of the likelihoods
                score = np.mean(likelihoods)
                if score > best_score:
                    best_score, best_model = score, model 

            except Exception as e:
                #logging.exception('CV exception occurred:', e)
                pass
        
        if not best_model:
            return self.base_model(self.n_constant)
        return best_model
