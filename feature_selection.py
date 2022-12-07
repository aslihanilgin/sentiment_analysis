import nltk.sentiment.vader as vd
from nltk.tokenize import word_tokenize
import nltk
import pandas as pd



class feature_selection:

    def __init__(self, number_classes):

        self.number_classes = number_classes

    def tag(self, sentence):
        # debug
        # import pdb; pdb.set_trace()
        # tokenize sentences
        sentence = word_tokenize(sentence)
        tagged_sentence = nltk.pos_tag(sentence) 
        new_sentence = list()
        # get all wanted tags
        tags = ('JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS')
        new_sentence.append([token for (token, tag) in tagged_sentence if tag.startswith(tags)])

        return new_sentence

    def negation(self, sentence):

        if vd.VaderConstants.negated(vd.VaderConstants, sentence, include_nt=True):
            # return vd.VaderConstants.N_SCALAR
            return 1
        else:
            return 0

    def intensifier(self, sentence):
        total_inc_dec_val = 0
        for token in sentence:
            if token in vd.VaderConstants.BOOSTER_DICT:
                # get increment/decrement value
                # TODO: should I add this value in neg post prob or post prob???
                # total_inc_dec_val += vd.VaderConstants.BOOSTER_DICT[token]
                total_inc_dec_val += 1
            else:
                continue

        return total_inc_dec_val

    # def exclamation():
        