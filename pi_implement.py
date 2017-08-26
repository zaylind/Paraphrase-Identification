# coding=utf-8
'''
@author: yidong zhang

@mail: yidong.zh@foxmail.com
'''

from numpy import *
from sklearn import metrics
from gensim import corpora, models
from scipy.linalg import norm
from sklearn import svm
from sklearn.linear_model import LogisticRegression
import random as rd
import time
import warnings
warnings.filterwarnings("ignore")

class ParaphraseIdentification(object):
    def __init__(self, train_file_path, test_file_path, algo="Random"):
        '''
        :param train_file_path:
        :param test_file_path:
        :param algo: algorithm selection: "Random", "Vector based", "MF_classify"
        '''
        train_index, train_pair, train_label, test_index, test_pair, test_label = \
            self._data_transform(train_file_path, test_file_path)
        self.algo = algo
        self.train_index = train_index
        self.train_pair = train_pair
        self.train_label = train_label
        self.test_index = test_index
        self.test_pair = test_pair
        self.test_label = test_label

    ### Read the dataset from files, and transform it to dict
    def _data_transform(self, train_file_path, test_file_path):
        fr_train = open(train_file_path)
        fr_test = open(test_file_path)
        train_index, train_pair, train_label = self._data_preprocess(fr_train)
        test_index, test_pair, test_label = self._data_preprocess(fr_test)
        return train_index, train_pair, train_label, test_index, test_pair, test_label

    def _data_preprocess(self, fr):
        lines = fr.readlines()
        str_pair = {}
        str_label = {}
        index_list = []
        for i, line in enumerate(lines):
            line = line.strip().split('	')
            if i == 0:
                continue
            else:
                index_str = '%s+%s' % (line[1], line[2])
                index_list.append(index_str)
                str_label[index_str] = int(line[0])
                str_pair.setdefault(index_str, [])
                str_pair[index_str].append(line[3])
                str_pair[index_str].append(line[4])
        return index_list, str_pair, str_label

    ### Obtain the corpus from dataset, including train data and test data
    def _get_corpus_list(self):
        documents = []
        train_index_count = 0
        for index, pair in self.train_pair.items():
            documents.append(pair[0])
            documents.append(pair[1])
            train_index_count += 2
        for index, pair in self.test_pair.items():
            documents.append(pair[0])
            documents.append(pair[1])
        stoplist = set('for a of the and to in on at'.split())
        texts = [[word for word in document.lower().split() if word not in stoplist]
                 for document in documents]
        dictionary = corpora.Dictionary(texts)
        # print "Number of dictionary: ", len(dictionary.keys())
        corpus = [dictionary.doc2bow(text) for text in texts]
        return corpus, train_index_count, len(dictionary.keys())

    ### Obtain the tf-idf model
    def _get_tfidf_dict(self, corpus):
        return models.TfidfModel(corpus)

    ### Use SVD to extract the tf-idf vector
    def _get_dict_2_matrix(self, corpus, tfidf_dict, dictionary_num, feature_num):
        matrix_set = []
        for value in corpus:
            vector_tmp = zeros((1, dictionary_num))[0].tolist()
            for k, v in dict(tfidf_dict[value]).items():
                vector_tmp[k] = v
            matrix_set.append(vector_tmp)
        U_mat, S_mat, v_mat = linalg.svd(array(matrix_set))
        reshape_matrix = U_mat[:, 0: feature_num]
        return reshape_matrix

    ### Get a feature vector from one sentence pair, which as a pre-processing for classifier
    ### concatenating the element-wise sum and absolute difference
    def _get_one_feature_matrix(self, vector_1, vector_2):
        feature_vector = (vector_1 + vector_2).tolist()
        feature_vector.extend((vector_1 - vector_2).tolist())
        return feature_vector
    
    ### Obtain the train data and test data with their feature for machine learning algorithm
    def _get_pair_feature(self, reshape_matrix, train_index_count):
        train_pair_index = [(i, i+1) for i in range(0, train_index_count, 2)]
        test_pair_index = [(i, i+1) for i in range(train_index_count, len(reshape_matrix), 2)]
        train_data = []
        train_label = []
        test_data= []
        test_label = []
        for (i, j) in train_pair_index:
            train_data.append(self._get_one_feature_matrix(reshape_matrix[i], reshape_matrix[j]))
        for index in self.train_index:
            train_label.append(self.train_label[index])
        for (i, j) in test_pair_index:
            test_data.append(self._get_one_feature_matrix(reshape_matrix[i], reshape_matrix[j]))
        for index in self.test_index:
            test_label.append(self.test_label[index])
        return train_data, train_label, test_data, test_label

    ### A randombaseline created by randomly choosing a true (paraphrase) 
    ### or false (not paraphrase) value for each text pair
    def _random_decision_function(self):
        predict_label = {}
        L_pred = []
        L_true = []
        for index in self.test_index:
            predict_label[index] = rd.randint(0, 1)
        for index in self.test_index:
            L_pred.append(predict_label[index])
            L_true.append(self.test_label[index])
        return L_pred, L_true

    ### A vector-based similarity baseline, using a cosine similarity
    # ### measure as traditionally used in information retrieval, with tf.idf weighting.
    def _vector_similarity_function(self):
        predict_label = []
        L_true = []
        corpus, train_index_count, dictionary_num = self._get_corpus_list()
        tf_idf_dict = self._get_tfidf_dict(corpus)
        samples_number = len(corpus)
        test_pair = [(i, i+1) for i in range(train_index_count, samples_number, 2)]
        for (p, q) in test_pair:
            vector_1 = tf_idf_dict[corpus[p]]
            vector_2 = tf_idf_dict[corpus[q]]
            similarity_value = self._get_cosine_sim(vector_1, vector_2)
            if similarity_value > 0.5:
                predict_label.append(1)
            else:
                predict_label.append(0)
        for index in self.test_index:
            L_true.append(self.test_label[index])
        return predict_label, L_true

    ### Cosine similarity function for dict
    def _get_cosine_sim(self, vector_1, vector_2):
        dict_1 = dict(vector_1)
        dict_2 = dict(vector_2)
        if len(dict_2) < len(dict_1):
            dict_1, dict_2 = dict_2, dict_1
        res = 0.0
        for key, dict_1_value in dict_1.iteritems():
            res += dict_1_value * dict_2.get(key, 0)
        if res == 0.0:
            return 0.0
        try:
            res = res / (norm(dict_1.values()) * norm(dict_2.values()))
        except ZeroDivisionError:
            res = 0.0
        return res

    def _classifier_decision_function(self, feature_num=100):
        predict_label = []
        '''
        corpus, train_index_count, dictionary_num = self._get_corpus_list()
        tf_idf_dict = self._get_tfidf_dict(corpus)
        reshape_matrix = self._get_dict_2_matrix(corpus, tf_idf_dict, dictionary_num, feature_num)
        train_data, train_label, test_data, test_label = self._get_pair_feature(reshape_matrix, train_index_count)
        save("./Data/train_data_100.npy", train_data)
        save("./Data/test_data_100.npy", test_data)
        '''
        train_data = load("./Data/train_data_100.npy")
        train_label = load("./Data/train_label.npy")
        test_data = load("./Data/test_data_100.npy")
        test_label = load("./Data/test_label.npy")
        # clf = svm.LinearSVC().fit(train_data, train_label)
        clf = LogisticRegression().fit(train_data, train_label)
        for i in range(len(test_data)):
            predict_label.append(clf.predict(test_data[i])[0])
        return predict_label, test_label

    def _measure_function(self, L_pred, L_true):
        precision = metrics.precision_score(L_true, L_pred)
        recall = metrics.recall_score(L_true, L_pred)
        f1_score = metrics.f1_score(L_true, L_pred)
        return precision, recall, f1_score

    ### Ensemble the flow process
    def pipeline(self):
        if self.algo == "Random":
            L_pred, L_true = self._random_decision_function()
        elif self.algo == "Vector based":
            L_pred, L_true = self._vector_similarity_function()
        elif self.algo == "MF_classify":
            L_pred, L_true = self._classifier_decision_function()
        else:
            raise Exception
        return self._measure_function(L_pred, L_true)

if __name__ == "__main__":
    print '%s:  Begin!' % (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    algorithm = "MF_classify"
    Pi = ParaphraseIdentification('./Data/msr_paraphrase_train.txt', './Data/msr_paraphrase_test.txt', algo=algorithm)
    precision, recall, f1_score = Pi.pipeline()
    #######################################################################################
    print "Precision value with %s" % algorithm, precision
    print "Recall value with %s" % algorithm, recall
    print "F1_score value with %s" % algorithm, f1_score
    print '%s:  End!' % (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))


#######################################################################################
'''
-----------------------------------------------------------------------------------------
# Random baseline #
Precision: 0.654205607477
Recall: 0.48823016565
F1_score: 0.559161258113
-----------------------------------------------------------------------------------------
# Vector-based similarity with Tf-idf weight baseline #
Precision: 0.650972364381
Recall: 0.554489973845
F1_score: 0.598870056497
-----------------------------------------------------------------------------------------
# Matrix factorization with Tf-idf weight classified by Logistic Regression #:
Precision: 0.664927536232
Recall: 1.0
F1_score: 0.798746518106
-----------------------------------------------------------------------------------------
# Matrix factorization with Tf-idf weight classified by Linear SVM #:
Precision: 0.663360560093
Recall: 0.991281604185
F1_score: 0.794826983572
-----------------------------------------------------------------------------------------
'''
#######################################################################################
