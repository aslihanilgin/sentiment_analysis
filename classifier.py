import numpy as np
import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class classifier:

    # Preprocesses sentences within the df by:
    # - lowercasing
    # - replacing "'s" with "is"
    def pre_process_sentences(self, df):

        for sentence in df["Phrase"]:
            # lowercase all phrases
            lower_sentences = sentence.lower()

            # remove punctuation
            rm_punc_sentence = re.sub(r'[^\w\s]','',lower_sentences)

            df['Phrase'] = df['Phrase'].replace([sentence], rm_punc_sentence)
        
        print("Preprocessed sentences.")
        return df
    
    def create_bag_of_words(self, df):

        print("Creating bag of words.")

        # Create a bag of words with counts for each 5 class
        all_words_and_counts = dict()

        for sentence in df["Phrase"]:

            # Reference: https://stackabuse.com/python-for-nlp-creating-bag-of-words-model-from-scratch/
            # tokenize sentences
            sentence_tokens = word_tokenize(sentence)
            sentence_tokens = sentence.split()

            # remove stop words
            # Reference: https://stackoverflow.com/questions/5486337/how-to-remove-stop-words-using-nltk-or-python
            sentence_tokens = [word for word in sentence_tokens if word not in stopwords.words('english')]

            for token in sentence_tokens:

                # get sentiment value of word
                sent_value = df[df['Phrase']==sentence]['Sentiment'].values[0] 

                # Reference: https://thispointer.com/python-dictionary-with-multiple-values-per-key/
                
                if token not in all_words_and_counts.keys(): # create a key with token if it doesn't exist in bag of words
                    all_words_and_counts[token] = list()
                    # initialise list with values 0 
                    all_words_and_counts[token] = [0] * 5 # -> [neg, sw_neg, neu, sw_pos, pos] 
                    all_words_and_counts[token][sent_value] = 1
                else:
                    # increment the count value
                    all_words_and_counts[token][sent_value] += 1
            
        print("Created bag of words.")
                    
        return all_words_and_counts

    def compute_total_sent_counts(self, df, number_classes):
        # get count of sentiments

        # Reference: https://sparkbyexamples.com/pandas/pandas-extract-column-value-based-on-another-column/#:~:text=You%20can%20extract%20a%20column,column%20value%20matches%20with%2025000.
        total_neg_word_count = len(df.query('Sentiment == 0')['Phrase'])
        total_sw_neg_word_count = len(df.query('Sentiment == 1')['Phrase'])
        total_neu_word_count = len(df.query('Sentiment == 2')['Phrase'])
        total_sw_pos_word_count = len(df.query('Sentiment == 3')['Phrase'])
        total_pos_word_count = len(df.query('Sentiment == 4')['Phrase'])

        if number_classes == 5:
            count_list = [total_neg_word_count, total_sw_neg_word_count, total_neu_word_count, total_sw_pos_word_count, total_pos_word_count]
        elif number_classes == 3:
            count_list = [(total_neg_word_count + total_sw_neg_word_count), total_neu_word_count, (total_sw_pos_word_count + total_pos_word_count)]
        
        return count_list

    def compute_prior_probability(self, total_sentence_no, count_list, number_classes):

        if number_classes == 5: 
            prior_prob_neg = count_list[0] / total_sentence_no
            prior_prob_sw_neg = count_list[1] / total_sentence_no
            prior_prob_neu = count_list[2] / total_sentence_no
            prior_prob_sw_pos = count_list[3] / total_sentence_no
            prior_prob_pos = count_list[4] / total_sentence_no
            
            return prior_prob_neg, prior_prob_sw_neg, prior_prob_neu, prior_prob_sw_pos, prior_prob_pos

        elif number_classes == 3: 
            prior_prob_neg = count_list[0] / total_sentence_no
            prior_prob_neu = count_list[1] / total_sentence_no
            prior_prob_pos = count_list[2] / total_sentence_no

            return prior_prob_neg, prior_prob_neu, prior_prob_pos

    # Compute likelihood for 5 sentiment values for a token
    def compute_likelihood_for_feature(self, token, sent_count_list, all_words_and_counts_dict, number_classes):

        if number_classes == 5:

            token_dict_vals = all_words_and_counts_dict[token] # list --> [neg, sw_neg, neu, sw_pos, pos]
            neg_c = token_dict_vals[0]
            sw_neg_c = token_dict_vals[1]
            neu_c = token_dict_vals[2]
            sw_pos_c = token_dict_vals[3]
            pos_c = token_dict_vals[4]

            neg_lh = neg_c / sent_count_list[0]
            sw_neg_lh = sw_neg_c / sent_count_list[1]
            neu_lh = neu_c / sent_count_list[2]
            sw_pos_lh = sw_pos_c / sent_count_list[3]
            pos_lh = pos_c / sent_count_list[4]

            return [neg_lh, sw_neg_lh, neu_lh, sw_pos_lh, pos_lh]   

        if number_classes == 3:

            token_dict_vals = all_words_and_counts_dict[token] # list --> [neg, neu, pos]
            neg_c = token_dict_vals[0] + token_dict_vals[1]
            neu_c = token_dict_vals[2]
            pos_c = token_dict_vals[3] + token_dict_vals[4]

            neg_lh = neg_c / sent_count_list[0]
            neu_lh = neu_c / sent_count_list[1]
            pos_lh = pos_c / sent_count_list[2]

            return [neg_lh, neu_lh, pos_lh]

    def compute_posterior_probability(self, sentence_lh_dict, class_prior_prob_list, number_classes):

        all_post_probs = list()

        if number_classes == 5:
            # TODO: can refactor it 
            sentence_neg_lh_prod = np.prod(np.array(sentence_lh_dict[0]), where=np.array(sentence_lh_dict[0])>0)
            sentence_sw_neg_lh_prod = np.prod(np.array(sentence_lh_dict[1]), where=np.array(sentence_lh_dict[1])>0)
            sentence_neu_lh_prod = np.prod(np.array(sentence_lh_dict[2]), where=np.array(sentence_lh_dict[2])>0)
            sentence_sw_pos_lh_prod = np.prod(np.array(sentence_lh_dict[3]), where=np.array(sentence_lh_dict[3])>0)
            sentence_pos_lh_prod = np.prod(np.array(sentence_lh_dict[4]), where=np.array(sentence_lh_dict[4])>0)

            neg_prior_prob = class_prior_prob_list[0]
            sw_neg_prior_prob = class_prior_prob_list[1]
            neu_prior_prob = class_prior_prob_list[2]
            sw_pos_prior_prob = class_prior_prob_list[3]
            pos_prior_prob = class_prior_prob_list[4]

            all_post_probs.append(sentence_neg_lh_prod * neg_prior_prob)
            all_post_probs.append(sentence_sw_neg_lh_prod * sw_neg_prior_prob)
            all_post_probs.append(sentence_neu_lh_prod * neu_prior_prob)
            all_post_probs.append(sentence_sw_pos_lh_prod * sw_pos_prior_prob)
            all_post_probs.append(sentence_pos_lh_prod * pos_prior_prob)
        

        elif number_classes == 3:
            sentence_neg_lh_prod = np.prod(np.array(sentence_lh_dict[0]), where=np.array(sentence_lh_dict[0])>0)
            sentence_neu_lh_prod = np.prod(np.array(sentence_lh_dict[1]), where=np.array(sentence_lh_dict[1])>0)
            sentence_pos_lh_prod = np.prod(np.array(sentence_lh_dict[2]), where=np.array(sentence_lh_dict[2])>0)

            neg_prior_prob = class_prior_prob_list[0]
            neu_prior_prob = class_prior_prob_list[1]
            pos_prior_prob = class_prior_prob_list[2]

            all_post_probs.append(sentence_neg_lh_prod * neg_prior_prob)
            all_post_probs.append(sentence_neu_lh_prod * neu_prior_prob)
            all_post_probs.append(sentence_pos_lh_prod * pos_prior_prob)

        highest_prob_index = np.argmax(all_post_probs) # returns index of highest probability score

        return highest_prob_index

    def count_sentiment_vals(self, df):

        total_sentence_no = len(df)

        total_neg_count = 0
        total_sw_neg_count = 0
        total_neu_count = 0
        total_sw_pos_count = 0
        total_pos_count = 0

        for sent_val in df["Sentiment"]:
            if sent_val == 0:
                neg_count += 1
            if sent_val == 1:
                sw_neg_count += 1
            if sent_val == 2:
                neu_count += 1
            if sent_val == 3:
                sw_pos_count += 1
            if sent_val == 4:
                pos_count += 1

        return total_sentence_no, total_neg_count, total_sw_neg_count, total_neu_count, total_sw_pos_count, total_pos_count

        