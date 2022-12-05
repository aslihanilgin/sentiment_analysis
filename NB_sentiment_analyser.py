# -*- coding: utf-8 -*-
"""
NB sentiment analyser. 

Start code.
"""
import argparse
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from f1_score_computation import f1_score_computation

"""
IMPORTANT, modify this part with your details
"""
USER_ID = "aca19aio"

def parse_args():
    parser=argparse.ArgumentParser(description="A Naive Bayes Sentiment Analyser for the Rotten Tomatoes Movie Reviews dataset")
    parser.add_argument("training")
    parser.add_argument("dev")
    parser.add_argument("test")
    parser.add_argument("-classes", type=int)
    parser.add_argument('-features', type=str, default="all_words", choices=["all_words", "features"])
    parser.add_argument('-output_files', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('-confusion_matrix', action=argparse.BooleanOptionalAction, default=False)
    args=parser.parse_args()
    return args

# Preprocesses sentences within the df by:
# - lowercasing
# - replacing "'s" with "is"
def pre_process_sentences(df):

    for sentence in df["Phrase"]:
        # lowercase all phrases, replace 's with is 
        lower_sentences = sentence.lower()
        lc_repl_sentence = lower_sentences.replace("'s", "is")
        # remove punctuation
        rm_punc_sentence = re.sub(r'[^\w\s]','',lc_repl_sentence)

        df['Phrase'] = df['Phrase'].replace([sentence], rm_punc_sentence)
    
    print("Just preprocessed sentences.")
    # debug
    print(df)
 
def create_bag_of_words(df):

    print("Going through each sentence now.")

    # Create a bag of words 
    global all_words_and_counts
    all_words_and_counts = dict()

    for sentence in df["Phrase"]:

        # Reference: https://stackabuse.com/python-for-nlp-creating-bag-of-words-model-from-scratch/
        # tokenize sentences
        sentence_tokens = word_tokenize(sentence)
        sentence_tokens = sentence.split()

        # remove stop words
        # Reference: https://stackoverflow.com/questions/5486337/how-to-remove-stop-words-using-nltk-or-python
        # tokens_wo_sw = [word for word in sentence_tokens if word not in stopwords.words('english')]

        for token in sentence_tokens:

            # get sentiment value of word
            sent_value = df[df['Phrase']==sentence]['Sentiment'].values[0] 

            # Reference: https://thispointer.com/python-dictionary-with-multiple-values-per-key/
            
            if token not in all_words_and_counts.keys(): # create a key with token if it doesn't exist in bag of words
                all_words_and_counts[token] = list()
                # initialise list with values 0 -> [neg, sw_neg, neu, sw_pos, pos] 
                all_words_and_counts[token] = [0] * 5 
                all_words_and_counts[token][sent_value] = 1
            else:
                # increment the count value
                all_words_and_counts[token][sent_value] += 1
                
    return all_words_and_counts

def compute_total_sent_counts(df, number_classes):
    # get count of sentiments

    # Reference: https://sparkbyexamples.com/pandas/pandas-extract-column-value-based-on-another-column/#:~:text=You%20can%20extract%20a%20column,column%20value%20matches%20with%2025000.
    total_neg_word_count = len(df.query('Sentiment == 0')['Phrase'])
    total_sw_neg_word_count = len(df.query('Sentiment == 1')['Phrase'])
    total_neu_word_count = len(df.query('Sentiment == 2')['Phrase'])
    total_sw_pos_word_count = len(df.query('Sentiment == 3')['Phrase'])
    total_pos_word_count = len(df.query('Sentiment == 4')['Phrase'])

    count_list = [total_neg_word_count, total_sw_neg_word_count, total_neu_word_count, total_sw_pos_word_count, total_pos_word_count]
    return count_list

def compute_prior_probability(total_sentence_no, count_list, number_classes):

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
def compute_likelihood_for_feature(token, sent_count_list, number_classes):

    if number_classes == 5:

        token_dict_vals = all_words_and_counts[token] # list --> [neg, sw_neg, neu, sw_pos, pos]
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

        token_dict_vals = all_words_and_counts[token] # list --> [neg, neu, pos]
        neg_c = token_dict_vals[0] + token_dict_vals[1]
        neu_c = token_dict_vals[2]
        pos_c = token_dict_vals[3] + token_dict_vals[4]

        neg_lh = neg_c / (sent_count_list[0] + sent_count_list[1])
        neu_lh = neu_c / sent_count_list[2]
        pos_lh = pos_c / (sent_count_list[3] + sent_count_list[4])

        return [neg_lh, neu_lh, pos_lh]

def compute_posterior_probability(sentence_lh_list, class_prior_prob_list, number_classes):

    all_post_probs = list()

    if number_classes == 5:
        # TODO: can refactor it 
        sentence_neg_lh = sentence_lh_list[0]
        sentence_sw_neg_lh = sentence_lh_list[1]
        sentence_neu_lh = sentence_lh_list[2]
        sentence_sw_pos_lh = sentence_lh_list[3]
        sentence_pos_lh = sentence_lh_list[4]

        neg_prior_prob = class_prior_prob_list[0]
        sw_neg_prior_prob = class_prior_prob_list[1]
        neu_prior_prob = class_prior_prob_list[2]
        sw_pos_prior_prob = class_prior_prob_list[3]
        pos_prior_prob = class_prior_prob_list[4]

        all_post_probs.append(sentence_neg_lh * neg_prior_prob)
        all_post_probs.append(sentence_sw_neg_lh * sw_neg_prior_prob)
        all_post_probs.append(sentence_neu_lh * neu_prior_prob)
        all_post_probs.append(sentence_sw_pos_lh * sw_pos_prior_prob)
        all_post_probs.append(sentence_pos_lh * pos_prior_prob)

    elif number_classes == 3:
        sentence_neg_lh = sentence_lh_list[0]
        sentence_neu_lh = sentence_lh_list[1]
        sentence_pos_lh = sentence_lh_list[2]

        neg_prior_prob = class_prior_prob_list[0]
        neu_prior_prob = class_prior_prob_list[1]
        pos_prior_prob = class_prior_prob_list[2]

        all_post_probs.append(sentence_neg_lh * neg_prior_prob)
        all_post_probs.append(sentence_neu_lh * neu_prior_prob)
        all_post_probs.append(sentence_pos_lh * pos_prior_prob)

    highest_prob_index = np.argmax(all_post_probs) # returns index of highest probability score

    return highest_prob_index

def count_sentiment_vals(df):

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



def main():
    
    inputs=parse_args()
    
    #input files
    training = inputs.training
    dev = inputs.dev
    test = inputs.test
    
    #number of classes
    number_classes = inputs.classes
    if not (number_classes == 3 or number_classes == 5):
        print("Number of classes specified is not applicable. Defaulting to 5.")
        number_classes = 5
    
    #accepted values "features" to use your features or "all_words" to use all words (default = all_words)
    features = inputs.features
    
    #whether to save the predictions for dev and test on files (default = no files)
    output_files = inputs.output_files
     
    
    #whether to print confusion matrix (default = no confusion matrix)
    confusion_matrix = inputs.confusion_matrix

    # read training tsv file
    df = pd.read_csv(training, delimiter = "\t")
    dev_df = pd.read_csv(dev, delimiter = "\t")

    # --------------- TRAINING DATA

    # preprocess dataframe
    pre_process_sentences(df)
  
    all_words_and_counts_dict = create_bag_of_words(df)

    ######

    # 1. Compute prior probability of each class
 
    # compute count for every class
    sent_count_list = compute_total_sent_counts(df, number_classes)
    
    # compute number of sentences
    total_sentence_no = len(df)

    # compute prior probabilities for each class
    if number_classes == 5:
        prior_prob_neg, prior_prob_sw_neg, prior_prob_neu, prior_prob_sw_pos, prior_prob_pos = compute_prior_probability(total_sentence_no, sent_count_list, number_classes)
        class_prior_prob_list = [prior_prob_neg, prior_prob_sw_neg, prior_prob_neu, prior_prob_sw_pos, prior_prob_pos]
    elif number_classes == 3:
        prior_prob_neg, prior_prob_neu, prior_prob_pos = compute_prior_probability(total_sentence_no, sent_count_list, number_classes)
        class_prior_prob_list = [prior_prob_neg, prior_prob_neu, prior_prob_pos]

    # 2. For each class:
    #   ▶ Compute likelihood of each feature
        # TODO: this will be for specific tokens for selected tokens

    likelihood_for_features_dict = dict()

    for token in all_words_and_counts_dict.keys():

        lh_list = compute_likelihood_for_feature(token, sent_count_list, number_classes)
        likelihood_for_features_dict[token] = lh_list

    # print("Likelihood for features dict: {}".format(likelihood_for_features_dict.keys()))

    pred_sentiment_value_dict = dict()

    # create sentence likelihood and prior prob list
    for sentence in df["Phrase"]: # TODO: CHECK IF df IS GENERALISED
        
        sentence_lh_list = list()

        word_list = sentence.split()

        for word in word_list:
            # class prior probability * every word's likelihood --> this is posterior probability
            for class_no in range(number_classes):
                sentence_lh_list.append(likelihood_for_features_dict[word][class_no])

        # 3. Calculate the posterior probability by product of previous components
        highest_prob_index = compute_posterior_probability(sentence_lh_list, class_prior_prob_list, number_classes)
    # 4. Select sentiment having maximum posterior probability
    #   ▶ negative, positive or neutral

        # add the sentence id and the calculated sent value to sentiment_value_dict
        sentence_id = df.loc[df['Phrase'] == sentence, 'SentenceId'].item() # TODO: CHECK IF df IS GENERALISED
        
        pred_sentiment_value_dict[sentence_id] = highest_prob_index

    ######

    # -----END---------- TRAINING DATA

    result = f1_score_computation(pred_sentiment_value_dict, df, number_classes)
    macro_f1_score = result.compute_macro_f1_score()
    
    # TODO: placeholder
    features = 0
    #You need to change this in order to return your macro-F1 score for the dev set

    """
    IMPORTANT: your code should return the lines below. 
    However, make sure you are also implementing a function to save the class predictions on dev and test sets as specified in the assignment handout
    """
    #print("Student\tNumber of classes\tFeatures\tmacro-F1(dev)\tAccuracy(dev)")
    print("%s\t%d\t%s\t%f" % (USER_ID, number_classes, features, macro_f1_score))

if __name__ == "__main__":
    main()