class f1_score_computation:
    def __init__(self, train_df, comp_df, number_classes):
        self.dev_file = train_df # may not need this
        self.compared_file = comp_df # may not need this
        self.number_classes = number_classes

        train_sent_and_id_df = train_df.loc[:, ["SentenceId", "Sentiment"]]
        self.train_sent_and_id_dict = dict(train_sent_and_id_df.values.tolist())

        comp_sent_and_id_df = comp_df.loc[:, ["SentenceId", "Sentiment"]]
        self.comp_sent_and_id_dict = dict(comp_sent_and_id_df.values.tolist())

        return self.compute_f1_score(number_classes)

    def compute_f1_score(self, number_classes):

        f1_scores = list()

        for class_id in range(number_classes):
            tp_count, fp_count, fn_count, tn_count = self.compute_counts(class_id)
            recall_val = self.compute_recall(class_id)
            precision_val = self.compute_precision(class_id)

            # computation of f1 score
            f1_score = 2 * ((precision_val * recall_val) / (precision_val + recall_val))

            f1_scores.append(f1_score)

        return f1_scores

    # def compute_macro_f1_score(self, f1_scores_of_classes):
        

    def compute_counts(self, class_id):

        tp_count = 0
        fp_count = 0
        fn_count = 0
        tn_count = 0

        for id in self.train_sent_and_id_dict.keys():
            if id in self.comp_sent_and_id_dict:
                # retrieved 
                if self.train_sent_and_id_dict[id] == class_id:
                    # true positives
                    if self.comp_sent_and_id_dict[id] == class_id:
                        tp_count += 1
                    # false positives
                    else:
                        fp_count += 1

                # not retrieved
                else: 
                    # false negatives
                    if self.comp_sent_and_id_dict[id] == class_id:
                        fn_count += 1
                    # true negatives
                    else: 
                        tn_count += 1


        return tp_count, fp_count, fn_count, tn_count
      
    
    def compute_recall(self, tp_count, fn_count):
        return tp_count / (tp_count + fn_count)


    def compute_precision(self, tp_count, fp_count):
        return tp_count / (tp_count + fp_count)


