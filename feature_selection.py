import nltk.sentiment.vader as vd
from nltk.tokenize import word_tokenize
import nltk

class feature_selection:

    def tag(self, sentence):
        # tokenize sentences
        sentence = word_tokenize(sentence)
        tagged_sentence = nltk.pos_tag(sentence) 
        new_sentence = list()
        # get all wanted tags
        tags = ('JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS')
        new_sentence.append([token for (token, tag) in tagged_sentence if tag.startswith(tags)])

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
        