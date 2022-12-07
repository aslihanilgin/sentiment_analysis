import numpy as np
import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from feature_selection import feature_selection


class classifier:

    def __init__(self, features):
        self.features = features

    # Preprocesses sentences within the df by:
    # - lowercasing
    # - removing stop words
    def pre_process_sentences(self, df):

        for sentence in df["Phrase"]:
            # lowercase all phrases
            lower_sentences = sentence.lower()

            # remove punctuation
            rm_punc_sentence = re.sub(r'[^\w\s]','',lower_sentences)

            # Reference: https://stackabuse.com/python-for-nlp-creating-bag-of-words-model-from-scratch/
            # tokenize sentences
            sentence_tokens = word_tokenize(rm_punc_sentence)

            # remove stop words
            # Reference: https://stackoverflow.com/questions/5486337/how-to-remove-stop-words-using-nltk-or-python
            sentence_tokens = [word for word in sentence_tokens if word not in stopwords.words('english')]
            rep_sentence = ' '.join(sentence_tokens)

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

            # if self.features == 'features': 

            #     feature_ops = feature_selection(self.features, number_classes) # create a feature selection object

            #     # change likelihoods according to adjectives present in sentence
            #     # lh_modification_vals = feature_ops.find_adjectives(sentence)

            #     # add negation sentiment value if there are any
            #     add_val_neg = (feature_ops.negation(sentence))

            #     # add intensifier value if there are any
            #     add_val_intens = (feature_ops.intensifier(sentence))

        highest_prob_index = np.argmax(all_post_probs) # returns index of highest probability score

        return highest_prob_index
        