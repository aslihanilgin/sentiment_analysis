import nltk.sentiment.vader as vd
from nltk.tokenize import word_tokenize
import nltk

from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np

class feature_selection:

    def tfidf(self, sentence_tokens, docs_counts):

        # Reference: https://sites.pitt.edu/~naraehan/presentation/Movie%20Reviews%20sentiment%20analysis%20with%20Scikit-Learn.html
        if len(sentence_tokens) > 0:

            sentence_freq = list()
            for token in sentence_tokens:
                if token in docs_counts:
                    token_freq = sum(docs_counts[token]) # how many times token appears in dataset
                    sentence_freq.append(token_freq)
                else: # token not in training data
                    sentence_freq.append(0)
            # debug
            # import pdb; pdb.set_trace()

            fooTfmer = TfidfTransformer()

            sentence_freq = np.array(sentence_freq).reshape(1, -1)
            docs_tfidf = fooTfmer.fit_transform(sentence_freq)
            feature_tfidf_array = docs_tfidf.toarray().tolist()[0]

            token_tfidf_dict = {sentence_tokens[i] : feature_tfidf_array[i] for i in range(len(feature_tfidf_array))}
            # Reference : https://stackoverflow.com/questions/613183/how-do-i-sort-a-dictionary-by-value
            sorted_token_tfidf_dict = dict(sorted(token_tfidf_dict.items(), key=lambda item: item[1]))

            number_of_features_to_ignore = round(len(sentence_tokens) / 5)

            # Reference: https://www.geeksforgeeks.org/python-remove-last-element-from-dictionary/#:~:text=Method%201%3A%20Using%20popitem(),last%20key%20from%20the%20dictionary.
            lowest_tfidf_keys = list(sorted_token_tfidf_dict)[:number_of_features_to_ignore]

            for key in lowest_tfidf_keys:
                if key not in sorted_token_tfidf_dict:
                    continue
                sorted_token_tfidf_dict.pop(key)

            return list(sorted_token_tfidf_dict.keys())

        else: # sentence is of length 0
            return None



    def tag(self, sentence):
        # Reference: https://stackoverflow.com/questions/15388831/what-are-all-possible-pos-tags-of-nltk
        tagged_sentence = nltk.pos_tag(sentence) 
        # get all wanted tags
        tags = ('JJ', 'JJR', 'JJS', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBP', 'VBZ')
        new_sentence = [token for (token, tag) in tagged_sentence if tag.startswith(tags)]

        return new_sentence

    # Reference: https://www.nltk.org/_modules/nltk/sentiment/vader.html
    def negation(self, token):

        if token in vd.VaderConstants.NEGATE:
            return vd.VaderConstants.N_SCALAR
            # return 1
        else:
            return 0

    # Reference: https://www.nltk.org/_modules/nltk/sentiment/vader.html
    def intensifier(self, token):

        if token in vd.VaderConstants.BOOSTER_DICT:
            # get increment/decrement value
            # TODO: should I add this value in neg post prob or post prob???
            # total_inc_dec_val += vd.VaderConstants.BOOSTER_DICT[token]
            return vd.VaderConstants.BOOSTER_DICT[token]
            # return 1

        else:
            return 0

    # def exclamation():
        
    