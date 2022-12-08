# Sentiment Analysis of Movie Reviews

## Packages Used

There is no additional installation needed to run this program.

The modules and their versions used for this project are as follows:

* pandas (Version: 1.4.2)
* nltk (Version: 3.7)
    * submodule: vader
    * submodule: corpus
    * submodule: stem
    * submodule: tokenize
* sklearn (Version: 1.0.2)
    * submodule: feature_extraction
* numpy (Version: 1.21.5)
* matplotlib (Version: 3.5.1)
* seaborn (Version: 0.11.2)

## Running the program

You need to be at the root directory `sentiment_analysis` of the project to run the program. There are a few options that can be provided with the run command.

You can run this program with the reuqired arguments as follows:

`python NB_sentiment_analyser.py <path to train file> <path to dev file> <path to test file>`

The optional arguments are:
* `-classes`: number of classes -> default **5** classes
* `-features`: features -> default is considering **all words**
* `-output_files`: option to display output files -> default is producing no output files
* `-confusion_matrix`: option to display the confusion matrix -> default is showing no confusion matrix

Since our train/dev/test datasets are in a directory called `moviereviews`, we will need to specify that when we are supplying the names of the input files. An example run command where we want to consider 3 classes with producing output files and confusion matrices for each class will be:

`python NB_sentiment_analyser.py moviereviews/train.tsv moviereviews/dev.tsv moviereviews/test.tsv -classes 3 -output_files -confusion_matrix`

If `output_files` argument is provided in the run command, the produced output files will be created in the `predictions` directory. 
