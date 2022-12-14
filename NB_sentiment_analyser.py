# -*- coding: utf-8 -*-
"""
NB sentiment analyser. 

Start code.
"""
import argparse
import csv 
import pandas as pd
from classifier import classifier
from f1_score_computation import f1_score_computation
from feature_selection import feature_selection

from nltk.tokenize import word_tokenize

"""Classifying sentiment value of movie review data.

Usage:
    `python NB_sentiment_analyser.py <path to train file> <path to dev file> <path to test file>`

    For more information, refer to README.md

Author:
    Aslihan Ilgin Okan - December 2022
"""
USER_ID = "aca19aio"

def parse_args():
    parser=argparse.ArgumentParser(description="A Naive Bayes Sentiment Analyser for the Rotten Tomatoes Movie Reviews dataset")
    parser.add_argument("training")
    parser.add_argument("dev")
    parser.add_argument("test")
    parser.add_argument("-classes", type=int)
    parser.add_argument('-features', type=str, default="all_words", choices=["all_words", "features_word_type", "features_tfidf"])
    parser.add_argument('-output_files', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('-confusion_matrix', action=argparse.BooleanOptionalAction, default=False)
    args=parser.parse_args()
    return args

def map_5_val_to_3_val_scale(df):

    df['Sentiment'] = df['Sentiment'].replace(1, 0)
    df['Sentiment'] = df['Sentiment'].replace(2, 1)
    df['Sentiment'] = df['Sentiment'].replace(3, 2)
    df['Sentiment'] = df['Sentiment'].replace(4, 2)

    return df

"""
Kickstart method for classification:
Pre-processes data, computes sentiment counts, computes 
prior probabilities, creates bag of words, computes
likelihoods

Returns: prior probability dict for each class, class likelihoods 
values dict for each unique term, frequency dict of each unique term
""" 
def start_classification(classification, train_df, number_classes, feature_opt):

    # preprocess dataframe
    train_df = classification.pre_process_sentences(train_df)

    # ----Computing Prior Probabilities
    
    # compute count for every sentiment class
    sent_count_list = classification.compute_total_sent_counts(train_df, number_classes)
    
    total_sentence_no = len(train_df) # compute total number of sentences

    # compute prior probabilities according to class number
    class_prior_prob_list = classification.compute_prior_probability(total_sentence_no, sent_count_list, number_classes)

    # -----Computing Likelihoods 

    likelihood_for_features_dict = dict()
    # create a bag of words with their counts
    all_words_and_counts_dict = classification.create_bag_of_words(train_df, number_classes)

    words_to_compute_lh = all_words_and_counts_dict.keys()

    if feature_opt == 'features_tfidf':
        # only choose relevant ones
        feature_ops = feature_selection()
        tfidf_selected_tokens = feature_ops.tfidf(all_words_and_counts_dict)
        words_to_compute_lh = tfidf_selected_tokens

    print("Computing likelihoods for features.")
    # compute likelihoods
    for token in words_to_compute_lh:
        likelihood_list = classification.compute_likelihood_for_feature(token, sent_count_list, all_words_and_counts_dict, number_classes)
        likelihood_for_features_dict[token] = likelihood_list
    
    return class_prior_prob_list, likelihood_for_features_dict, all_words_and_counts_dict

"""
Evaluates given file:
Pre-processes to be evaluated data, calculates posterior 
probabilities, predicts sentiment values

Returns: sentence id with its predicted sentiment value dictionary
"""   
def evaluate_file(classification, eval_df, class_prior_prob_list, likelihood_for_features_dict, number_classes, feature_opt, all_words_and_counts_dict):

    print("Evaluating file.")

    # preprocess dataframe
    eval_df = classification.pre_process_sentences(eval_df)

    pred_sentiment_value_dict = dict()

    # Calculate posterior probability for every sentence in dev file
    loop_count = 0 
    for sentence in eval_df["Phrase"]:
        # Reference: https://stackabuse.com/python-for-nlp-creating-bag-of-words-model-from-scratch/
        # tokenize sentences
        sentence_tokens = word_tokenize(sentence)

        # Reference: https://www.programiz.com/python-programming/methods/dictionary/fromkeys
        sentence_lh_dict = { key : list() for key in range(number_classes)}
        for token in sentence_tokens:
            if token in likelihood_for_features_dict:
                for class_no in range(number_classes):
                    sentence_lh_dict[class_no].append(likelihood_for_features_dict[token][class_no])
            else: # token not in training bag of words
                continue

        # get sentiment having maximum posterior probability
        highest_prob_index = classification.compute_posterior_probability(sentence, sentence_lh_dict, class_prior_prob_list, number_classes)

        # add the sentence id and the calculated sent value to sentiment_value_dict
        sentence_id = eval_df.iloc[[loop_count]]['SentenceId'].item()
        
        pred_sentiment_value_dict[sentence_id] = highest_prob_index

        loop_count += 1
    
    print("Evaluation finished.")

    return pred_sentiment_value_dict

# Produces output file 
def produce_output_file(type, number_classes, user_id, pred_file):
    dir_path = "predictions/"
    file_name = dir_path + type + "_predictions_" + str(number_classes) + "classes_" + user_id + ".tsv"

    columns = ['SentenceID', 'Sentiment']

    pred_file_df = pd.DataFrame(list(pred_file.items()))
    pred_file_df.columns = columns
    
    pred_file_df.to_csv(file_name, sep="\t")

def main():
    
    inputs=parse_args()
    
    #input files
    training = inputs.training
    dev = inputs.dev
    test = inputs.test
    
    #number of classes
    number_classes = inputs.classes
    
    #accepted values "features" to use your features or "all_words" to use all words (default = all_words)
    features = inputs.features
    
    #whether to save the predictions for dev and test on files (default = no files)
    output_files = inputs.output_files
     
    
    #whether to print confusion matrix (default = no confusion matrix)
    confusion_matrix = inputs.confusion_matrix

    # read tsv files
    train_df = pd.read_csv(training, delimiter = "\t")
    dev_df = pd.read_csv(dev, delimiter = "\t")
    test_df = pd.read_csv(test, delimiter = "\t")

    if number_classes == 3:
        train_df = map_5_val_to_3_val_scale(train_df)
        dev_df = map_5_val_to_3_val_scale(dev_df)

    print("Read all files.")

    # create a classifier object
    classification = classifier(features)

    # kickstart classification
    print("Starting classification.")
    class_prior_prob_list, likelihood_for_features_dict, all_words_and_counts_dict = start_classification(classification, train_df, number_classes, features)

    # evaluate dev file
    dev_pred_sentiment_value_dict = evaluate_file(classification, dev_df, class_prior_prob_list, likelihood_for_features_dict, number_classes, features, all_words_and_counts_dict)
    f1_score_comp = f1_score_computation(dev_pred_sentiment_value_dict, dev_df, number_classes, confusion_matrix) # compare pred dev vs actual dev
    dev_macro_f1_score = f1_score_comp.compute_macro_f1_score()

    # evaluate test file
    test_pred_sentiment_value_dict = evaluate_file(classification, test_df, class_prior_prob_list, likelihood_for_features_dict, number_classes, features, all_words_and_counts_dict)

    if output_files:
        # write to output files
        produce_output_file('dev', number_classes, USER_ID, dev_pred_sentiment_value_dict)
        produce_output_file('test', number_classes, USER_ID, test_pred_sentiment_value_dict)


    print("Student\tNumber of classes\tFeatures\tmacro-F1(dev)\tAccuracy(dev)")
    print("%s\t%d\t%s\t%f" % (USER_ID, number_classes, features, dev_macro_f1_score))

if __name__ == "__main__":
    main()