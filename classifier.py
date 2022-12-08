import numpy as np
import re
import nltk.sentiment.vader as vd

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from feature_selection import feature_selection


class classifier:

    def __init__(self, features):
        self.features = features
        self.feature_ops = feature_selection()

    # Preprocesses sentences within the df by:
    # - lowercasing
    # - removing stop words
    def pre_process_sentences(self, df):

        for sentence in df["Phrase"]:

            # lowercase all phrases
            lower_sentence = sentence.lower()

            # remove punctuation
            rm_punc_sentence = re.sub(r'[^\w\s]','',lower_sentence)
            # remove numbers
            rm_num_sentence = re.sub(r'[0-9]', '', rm_punc_sentence)
            # replace nt with not
            repl_sentence = rm_num_sentence.replace("nt", "not")

            # Reference: https://stackabuse.com/python-for-nlp-creating-bag-of-words-model-from-scratch/
            # tokenize sentences
            sentence_tokens = word_tokenize(repl_sentence)

            # remove stop words
            # Reference: https://stackoverflow.com/questions/5486337/how-to-remove-stop-words-using-nltk-or-python
            sentence_tokens = [word for word in sentence_tokens if (word not in stopwords.words('english')) or (word not in vd.VaderConstants.NEGATE) or (word not in vd.VaderConstants.BOOSTER_DICT)]

            if self.features == 'features':
                sentence_tokens = self.feature_ops.tag(sentence_tokens)

            # stemming
            ps = PorterStemmer()
            stemmed_sentence_tokens = [ps.stem(t) for t in sentence_tokens if (t not in vd.VaderConstants.NEGATE) or (t not in vd.VaderConstants.BOOSTER_DICT)]
            rep_sentence = ' '.join(stemmed_sentence_tokens)

            df['Phrase'] = df['Phrase'].replace([sentence], rep_sentence)
        
        print("Preprocessed sentences.")
        # debug
        print(df)

        return df
    
    def create_bag_of_words(self, df, number_classes):

        # TODO: CAN DO THIS IN THE PREPROCESSING STEP TO AVOID ANOTHER L=OOP THROUGH DATAFRAME

        print("Creating bag of words.")

        # Create a bag of words with counts for each class
        all_words_and_counts = dict()

        for sentence in df["Phrase"]:

            # # Reference: https://stackabuse.com/python-for-nlp-creating-bag-of-words-model-from-scratch/
            # tokenize sentences
            sentence_tokens = word_tokenize(sentence)

            for token in sentence_tokens:

                # get sentiment value of word
                sent_value = df[df['Phrase']==sentence]['Sentiment'].values[0] 

                # Reference: https://thispointer.com/python-dictionary-with-multiple-values-per-key/
                
                if token not in all_words_and_counts.keys(): # create a key with token if it doesn't exist in bag of words
                    all_words_and_counts[token] = list()
                    # initialise list with values 0 
                    all_words_and_counts[token] = [0] * number_classes # -> [neg, sw_neg, neu, sw_pos, pos] 
                    all_words_and_counts[token][sent_value] = 1
                else:
                    # increment the count value
                    all_words_and_counts[token][sent_value] += 1

        print("Created bag of words.")
                    
        return all_words_and_counts

    def compute_total_sent_counts(self, df, number_classes):
        # get count of sentiments

        count_list = list()

        for class_no in range(number_classes):
            count = len(df.loc[df['Sentiment'] == class_no, ['Phrase']])

            count_list.append(count)

        return count_list

    def compute_prior_probability(self, total_sentence_no, count_list, number_classes):

        class_prior_probs = list()

        for class_no in range(number_classes):
            prior_prob = count_list[class_no] / total_sentence_no
            class_prior_probs.append(prior_prob)

        return class_prior_probs

    # Compute likelihood for 5 sentiment values for a token
    def compute_likelihood_for_feature(self, token, sent_count_list, all_words_and_counts_dict, number_classes):
        
        # list --> [neg, sw_neg, neu, sw_pos, pos] if 5 class
        # list --> [neg, neu, pos] if 3 class
        token_dict_vals = all_words_and_counts_dict[token] 

        likelihood_list = list()

        for class_no in range(number_classes):
            count = token_dict_vals[class_no]
            class_likelihood = count / sent_count_list[class_no]
            likelihood_list.append(class_likelihood)

        return likelihood_list


    def compute_posterior_probability(self, sentence, sentence_lh_dict, class_prior_prob_list, number_classes):

        all_post_probs = list()

        for class_no in range(number_classes):
            class_lh_product = np.prod(np.array(sentence_lh_dict[class_no]), where=np.array(sentence_lh_dict[class_no])>0)
            class_prior_prob = class_prior_prob_list[class_no]

            all_post_probs.append(class_lh_product * class_prior_prob)

        if self.features == 'features':
            neg_add_val = self.feature_ops.negation(sentence)
            intense_add_val = self.feature_ops.intensifier(sentence)

            if neg_add_val != None:
                all_post_probs[0] += neg_add_val

            if number_classes == 5:
                if neg_add_val != None:
                    all_post_probs[1] += neg_add_val
                if intense_add_val != None:
                    all_post_probs[3] += intense_add_val
                    all_post_probs[4] += intense_add_val
            if number_classes == 3:
                if intense_add_val != None:
                    all_post_probs[2] += intense_add_val

        highest_prob_index = np.argmax(all_post_probs) # returns index of highest probability score

        return highest_prob_index


        
        