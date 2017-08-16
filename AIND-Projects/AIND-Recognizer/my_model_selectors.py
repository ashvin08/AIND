import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


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
        #L = Likelihood of fitted model
        #p = Number of parameters
        #p =  = n^2 + 2*d*n - 1
        #N = Number of data points
        bic_score = float("inf")
        best_num_of_states = None
        try:
            for states in range(self.min_n_components, self.max_n_components + 1):
                model = self.base_model(states)
                logL = model.score(self.X, self.lengths)
                ##http://hmmlearn.readthedocs.io/en/latest/api.html#gaussianhmm
                p = states ** 2 + states*2*model.n_features - 1
                currect_bic_score = -2 * logL + p * math.log(model.n_features)
                #lower the score better the model
                if currect_bic_score < bic_score:
                    bic_score = currect_bic_score
                    best_num_of_states = states
        except:
            pass

        return self.base_model(best_num_of_states)


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))

    M = Set of categories or classes => Number of words in the topology
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        dic_score = float("-inf")
        best_num_of_states = None
        logLList = []
        m = len(self.words)
        for states in range(self.min_n_components, self.max_n_components + 1):
            sum_of_logL = 0
            try:
                model = self.base_model(states)
                logL = model.score(self.X, self.lengths)
                logLList.append(logL)
                sum_of_logL += logL
                for idx, logL in enumerate(logLList):
                    anti_evidence = (sum_of_logL - logL) / (m - 1)
                    current_dic_score = logL - anti_evidence
                    if current_dic_score > dic_score:
                        dic_score = current_dic_score
                        best_num_of_states = self.min_n_components + idx
            except:
                pass

        return self.base_model(best_num_of_states)


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        best_mean = float("-inf")
        best_num_of_states = None
        split_method = KFold(n_splits=3, shuffle = False, random_state = None)


        for states in range(self.min_n_components, self.max_n_components + 1):
            try:
                #Can the data be split into folds
                if len(self.sequences) > 2:
                    fold_scores = []
                    for train_idx, test_idx in split_method.split(self.sequences):
                        self.X, self.lengths = combine_sequences(train_idx, self.sequences)
                        X, length = combine_sequences(test_idx, self.sequences)

                        # train base model using training data
                        model = self.base_model(states)
                        #log likelihood is calculated against test data
                        logL = model.score(X, length)
                        fold_scores.append(logL)
                    #Calculate mean for the current state
                    current_mean = np.mean(fold_scores)
                # Use the base model log likelihood
                else:
                    model = self.base_model(states)
                    current_mean = model.score(self.X, self.lengths)

                #Calculate the best state
                if current_mean > best_mean:
                    best_mean = current_mean
                    best_num_of_states = states

            except:
                pass

        return self.base_model(best_num_of_states)
