# -*- coding: utf-8 -*-
"""
NB sentiment analyser. 

Start code.
"""
import argparse
import pandas as pd
import re
from nltk.corpus import stopwords
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

    # read tsv file
    df = pd.read_csv(training, delimiter = "\t")

    
    for sentence in df["Phrase"]:
        # lowercase all phrases, replace 's with is 
        lc_repl_sentence = sentence.lower().replace("'s", "is")
        # remove punctuation
        rm_punc_sentence = re.sub(r'[^\w\s]','',lc_repl_sentence)
        # tokenize sentences
        sentence_tokens = word_tokenize(rm_punc_sentence)

        # remove stopwords TODO: do I want the stopwords to calculate sentiment value of sentence? 
        # Reference: https://stackabuse.com/removing-stop-words-from-strings-in-python/
        # tokens_wo_sw = [word for word in sentence_tokens if not word in stopwords.words()]

        # debug
        # df['Phrase'] = df['Phrase'].replace(sentence, back_to_sentence)
        print(sentence_tokens)
  
        

   

    # remove stop words 
        
    print(df)

   

    
    # TODO: placeholder
    number_classes = 0
    features = 0
    #You need to change this in order to return your macro-F1 score for the dev set
    f1_score = 0
    

    """
    IMPORTANT: your code should return the lines below. 
    However, make sure you are also implementing a function to save the class predictions on dev and test sets as specified in the assignment handout
    """
    #print("Student\tNumber of classes\tFeatures\tmacro-F1(dev)\tAccuracy(dev)")
    print("%s\t%d\t%s\t%f" % (USER_ID, number_classes, features, f1_score))

if __name__ == "__main__":
    main()