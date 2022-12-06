import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class f1_score_computation:
    def __init__(self, pred_vals_dict, df, number_classes):
        self.number_classes = number_classes
        self.pred_vals_dict = pred_vals_dict

        sentiment_and_id_df = df.loc[:, ["SentenceId", "Sentiment"]]
        self.real_sentiment_values_dict = dict(sentiment_and_id_df.values.tolist())

    def compute_f1_scores(self):

        f1_scores = list()

        for class_id in range(self.number_classes):
            tp_count, fp_count, fn_count, tn_count = self.compute_counts(class_id)

            self.visualise_confusion_matrix(tp_count, fp_count, fn_count, tn_count)

            recall_val = self.compute_recall(tp_count, fn_count)
            precision_val = self.compute_precision(tp_count, fp_count)

            # computation of f1 score
            if (precision_val + recall_val) == 0:
                f1_score = 0
            else:
                f1_score = 2 * ((precision_val * recall_val) / (precision_val + recall_val))

            f1_scores.append(f1_score)

        return f1_scores

    def compute_macro_f1_score(self):

        f1_scores_of_classes = self.compute_f1_scores()

        macro_f1 = sum(f1_scores_of_classes) / self.number_classes
        return macro_f1

    def compute_counts(self, class_id):

        #debug
        # import pdb; pdb.set_trace()

        tp_count = 0
        fp_count = 0
        fn_count = 0
        tn_count = 0

        for id in self.pred_vals_dict.keys():
            if id in self.real_sentiment_values_dict:
                # retrieved 
                if self.pred_vals_dict[id] == class_id:
                    # true positives
                    if self.real_sentiment_values_dict[id] == class_id:
                        tp_count += 1
                    # false positives
                    else:
                        fp_count += 1

                # not retrieved
                else: 
                    # false negatives
                    if self.real_sentiment_values_dict[id] == class_id:
                        fn_count += 1
                    # true negatives
                    else: 
                        tn_count += 1


        return tp_count, fp_count, fn_count, tn_count
      
    
    def compute_recall(self, tp_count, fn_count):
        if (tp_count + fn_count) == 0:
            return 0
        else:
            return tp_count / (tp_count + fn_count)


    def compute_precision(self, tp_count, fp_count):
        if (tp_count + fp_count) == 0:
            return 0
        else:
            return tp_count / (tp_count + fp_count)

    def visualise_confusion_matrix(self, tp_count, fp_count, fn_count, tn_count):
        # Reference: https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea
        cf_matrix = [[tp_count, fp_count], [fn_count, tn_count]]
        df_cm = pd.DataFrame(cf_matrix,  ["Retrieved", "Not Retrieved"], ["Relevant", "Non-relevant"])
        sns.heatmap(df_cm, annot=True, fmt='.1f')

        plt.show()


