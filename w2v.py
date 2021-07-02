from gensim.models import Word2Vec as w2v
from sklearn import svm
import nltk
import pandas
import numpy as np
from sklearn import svm
import csv
import re
import pickle

class File_Tools():
    @staticmethod
    def get_tsv_col(file_name : str, *cols) -> dict:
        result = {}
        with open(file_name) as f:
            rd = csv.reader(f, delimiter="\t", quotechar='"')
            title_row = next(rd)
            rows = {}
            for col in cols:
                if col not in title_row:
                    Exception("Invalid column name")
                else:
                    rows[title_row.index(col)] = col 
            result = {row : [] for row  in rows.values()}
            for row in rd:
                for val in rows.keys():
                    result[rows[val]].append(row[val])
        return result
    
    @staticmethod
    def get_all_lines(file_name : str):
        res = []
        with open(file_name, "r", encoding="utf-8") as f:
            for line in f.read().split("\n"):
                res.append(line)
        return res
    
    @staticmethod
    def save_list_to_file(file_name : str, ls : list):
        with open(file_name, "w+", encoding="utf-8") as f:
            for phrase in ls:
                f.writelines(phrase + "\n")



class Text_Preprocessing():
    @staticmethod   
    def clean_phrases(phrases : list):
        res = []
        for phrase in phrases:
            phrase = phrase.lower().strip()
            phrase = re.sub(r'[^a-z0-9\s]', '', phrase)
            res.append(re.sub(r'\s{2,}', ' ', phrase))
        return res

#     @staticmethod
#     async def clean_phrases(phrases : list):
#         res = []
#         executor = concurrent.futures.ProcessPoolExecutor(10)
#         res = [executor.submit(Text_Preprocessing.__clean_phrases__, phrase_group) for phrase_group in grouper(10000, phrases)]
#         concurrent.futures.wait(res)
#         return res

#     @staticmethod
#     def tokenize(phrase):
#         return [token for token in phrase.split() if token not in STOP_WORDS]

class Word_To_Vec_Tools(object):
    def __init__(self) -> None:
        super().__init__()

    def make_corpus(self, input_file : str, col_name : str):
        if not col_name:
            self.__make_corpus__(input_file)
            return
        csv = pandas.read_csv(input_file)
        df = pandas.DataFrame(csv)
        sentences = df[col_name].values
        self.word_vec = [nltk.word_tokenize(sentence) for sentence in sentences]

    def __make_corpus__(self, input_file : str):
        sentences = File_Tools.get_all_lines(input_file)
        self.word_vec = [nltk.word_tokenize(sentence) for sentence in sentences]

    def load_model(self, model_name : str):
        self.model = w2v.load(model_name)

    def save_model(self, model_name : str):
        if not self.model:
            Exception("No model to save")
        self.model.save(model_name)

    def train_model(self, sg = 1, size = 300, window = 10, min_count = 2, workers = 4) -> None:
        if not self.word_vec:
            Exception("Must make word vec")
        self.model = w2v(self.word_vec, vector_size=size, window=window, epochs=10, sg=sg, min_count=min_count,workers=workers)
        #self.model.save("recent_model.model")

    def get_similar_words(self, word : str, num_words = 10) -> list:
        if not self.model:
            Exception("Must create model first")
        return self.model.most_simlar(word, size = num_words)

    # def compare_words(self, word1 : str, word2 : str):
    #     try:
    #         np.linalg.norm(self.model.wv[word1], self.model.wv[word2])
    #     except:
    #         print("words not present in w2v model")

    def get_average_of_sentence(self, phrase) -> list:
        words = nltk.word_tokenize(phrase)
        total = []
        if len(words) == 0:
            print(phrase)
            return -1
        for word in words:
            if word in self.model.wv:
                total.append(self.model.wv[word])
        return total

    def get_average_of_sentences(self, phrases) -> list:
        res = []
        for phrase in phrases:
            tmp = self.get_average_of_sentence(phrase)
            if type(tmp) != type([]):
                continue
            else:
                res.append(tmp)
        return res
    
    def get_similar(self, positive : list, negative : list):
        if not self.model:
            Exception("Must create model first")
        return self.model.wv.most_similar(positive = positive, negative = negative)



class Classifier():
    
    def __init__(self) -> None:
        pass
    
    def train_svm_classifier(self, phrases, categories) -> None:
        self.classifier = svm.SVC(kernel="poly", degree=6)
        self.classifier.fit(phrases, categories)
    
    def save_classifier(self, classifier_name) -> None:
        with open(classifier_name, 'wb') as f:
            pickle.dump(self.classifier, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def load_classifier(self, file_name) -> None:
        with open(file_name, 'rb') as f:
            self.classifier = pickle.load(f)
        
    def test_single_phrase_vector(self, phrase_vec):
        return self.classifier.predict([phrase_vec])
    
    def test_many_phrase_vectors(self, phrase_vecs):
        return self.classifier.predict(phrase_vecs)
    
    def test_results(self, phrase_vecs, res):
        return self.classifier.score(phrase_vecs, res)
    # def check_category(self, categories : list, phrase : str):
    #     sentence_score = self.get_average_of_sentence(phrase)
    #     max_similarity = 0
    #     res = None
    #     for category in categories:
    #         cos_similarity = np.dot(sentence_score, self.model.wv[category])/(np.linalg.norm(sentence_score) * np.linalg.norm(self.model.wv[category]))
    #         if cos_similarity > max_similarity:
    #             max_similarity = cos_similarity
    #             res = category
    #     return res 




def main():
    # w2vt = Word_To_Vec_Tools()
    # t = File_Tools.get_tsv_col("train.tsv", "Phrase")
    # File_Tools.save_list_to_file("Movie_Phrases.txt", t["Phrase"])
    # w2vt.make_corpus("Movie_Phrases.txt", None)
    # w2vt.train_model()
    # w2vt.save_model("movie_model.model")
    # sent = File_Tools.get_tsv_col("train.tsv", "Sentiment", "Phrase")
    # sent_avg = w2vt.get_average_of_sentences(sent["Phrase"])
    # k_v_pairs = []
    # for i in range(len(sent_avg)):
    #     try:
    #         #
    #         k_v_pairs.append((sent_avg[i][0], sent["Sentiment"][i]))
    #     except:
    #         pass
    # c = Classifier()
    # c.train_svm_classifier([i[0] for i in k_v_pairs], [j[1] for j in k_v_pairs])
    # c.save_classifier('svm_model.pickle')
    test_info = File_Tools.get_tsv_col("test.tsv", "Phrase", "PhraseId")
    w2vt = Word_To_Vec_Tools()
    w2vt.load_model("movie_model.model")
    sent_avg = w2vt.get_average_of_sentences(test_info["Phrase"])
    res = {}
    c = Classifier()
    c.load_classifier("svm_model.pickle")
    i = 0
    ls = []

    for i, p in enumerate(sent_avg):
        try:
            res[test_info["PhraseId"][i]] = c.test_single_phrase_vector(p[0])[0]
        except Exception as e:
            res[test_info["PhraseId"][i]] = 0
        i += 1
        if i % 1000 == 0:
            print(i)
    #Word_To_Vec_Tools.make_corpus()
   # print(tmp)
    with open ("Results.csv", "w", newline='') as f:
        w = csv.writer(f)
        w.writerow(["PhraseId", "Sentiment"])
        for k in res.keys():
            w.writerow([k, res[k]])


if __name__ == '__main__':
    main()
