# -*- coding: utf-8 -*-
"""
NB sentiment analyser. 

Start code.
"""
import argparse
import pandas as pd
from classifier import classifier
from f1_score_computation import f1_score_computation

from nltk.tokenize import word_tokenize

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

def start_classification(classification, train_df, number_classes):

    # preprocess dataframe
    train_df = classification.pre_process_sentences(train_df)

    # 1. Compute prior probability of each class
 
    # compute count for every sentiment class
    sent_count_list = classification.compute_total_sent_counts(train_df, number_classes)
    
    total_sentence_no = len(train_df) # compute total number of sentences

    # compute prior probabilities according to class number
    if number_classes == 5:
        prior_prob_neg, prior_prob_sw_neg, prior_prob_neu, prior_prob_sw_pos, prior_prob_pos = classification.compute_prior_probability(total_sentence_no, sent_count_list, number_classes)
        class_prior_prob_list = [prior_prob_neg, prior_prob_sw_neg, prior_prob_neu, prior_prob_sw_pos, prior_prob_pos]
    elif number_classes == 3:
        prior_prob_neg, prior_prob_neu, prior_prob_pos = classification.compute_prior_probability(total_sentence_no, sent_count_list, number_classes)
        class_prior_prob_list = [prior_prob_neg, prior_prob_neu, prior_prob_pos]

    # 2. For each class:
    #   ▶ Compute likelihood of each feature
        # TODO: this will be for specific tokens for selected tokens

    likelihood_for_features_dict = dict()
    # create a bag of words with their counts
    all_words_and_counts_dict = classification.create_bag_of_words(train_df)

    print("Computing likelihoods for features.")
    for token in all_words_and_counts_dict.keys():

        lh_list = classification.compute_likelihood_for_feature(token, sent_count_list, all_words_and_counts_dict, number_classes)
        likelihood_for_features_dict[token] = lh_list
    
    return class_prior_prob_list, likelihood_for_features_dict 

    # training done
    
    
def evaluate_dev(classification, dev_df, class_prior_prob_list, likelihood_for_features_dict, number_classes):

    print("Evaluating dev file.")

    # preprocess dataframe
    dev_df = classification.pre_process_sentences(dev_df)

    pred_sentiment_value_dict = dict()

    # Calculate posterior probability for every sentence in dev file
    for sentence in dev_df["Phrase"]:
        # Reference: https://stackabuse.com/python-for-nlp-creating-bag-of-words-model-from-scratch/
        # tokenize sentences
        sentence_tokens = word_tokenize(sentence)
        sentence_tokens = sentence.split()

        # Reference: https://www.programiz.com/python-programming/methods/dictionary/fromkeys
        sentence_lh_dict = { key : list() for key in range(number_classes)}
        # get likelihood list of every word in every sentence of dev data
        for token in sentence_tokens:
            if token in likelihood_for_features_dict:
                for class_no in range(number_classes):
                    sentence_lh_dict[class_no].append(likelihood_for_features_dict[token][class_no]) 
            else: # token not in training bag of words
                continue
        # get sentiment having maximum posterior probability
        highest_prob_index = classification.compute_posterior_probability(sentence_lh_dict, class_prior_prob_list, number_classes)

        # add the sentence id and the calculated sent value to sentiment_value_dict
        sentence_id = dev_df.loc[dev_df['Phrase'] == sentence, 'SentenceId'].item() # TODO: CHECK IF df IS GENERALISED
        
        pred_sentiment_value_dict[sentence_id] = highest_prob_index

    result = f1_score_computation(pred_sentiment_value_dict, dev_df, number_classes) # compare pred dev vs actual dev
    dev_macro_f1_score = result.compute_macro_f1_score()

    print("Evaluation finished.")

    return dev_macro_f1_score

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

    # read tsv files
    train_df = pd.read_csv(training, delimiter = "\t")
    dev_df = pd.read_csv(dev, delimiter = "\t")
    test_df = pd.read_csv(test, delimiter = "\t")

    print("Read all files.")

    # create a classifier object
    classification = classifier()

    # kickstart classification
    print("Starting classification.")
    class_prior_prob_list, likelihood_for_features_dict = start_classification(classification, train_df, number_classes)

    # evaluate dev file
    dev_macro_f1_score = evaluate_dev(classification, dev_df, class_prior_prob_list, likelihood_for_features_dict, number_classes)
    print("Dev macro f1 score: {}".format(dev_macro_f1_score))

    # # TODO
    # # evaluate test file
    # evaluate_test()

    # use the training data to get all calculations etc and apply it on the dev data


    # TODO: placeholder
    macro_f1_score = 0
    #You need to change this in order to return your macro-F1 score for the dev set

    """
    IMPORTANT: your code should return the lines below. 
    However, make sure you are also implementing a function to save the class predictions on dev and test sets as specified in the assignment handout
    """
    # print("Student\tNumber of classes\tFeatures\tmacro-F1(dev)\tAccuracy(dev)")
    print("%s\t%d\t%s\t%f" % (USER_ID, number_classes, features, macro_f1_score))

if __name__ == "__main__":
    main()