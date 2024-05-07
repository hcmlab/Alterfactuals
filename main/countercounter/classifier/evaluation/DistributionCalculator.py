import pandas as pd

from main.countercounter.classifier.evaluation.activationdistribution.feature_contribution import \
    return_feature_contribution_data, return_feature_contribution_data_with_identifier


class DistributionCalculator:

    def __init__(self, data_iterable, cnn):
        self.data_iterable = data_iterable
        self.cnn = cnn

    def calculate(self):
        activations_by_class, logits_by_class = return_feature_contribution_data(self.data_iterable, self.cnn, num_classes=2)

        feature_count = len(activations_by_class[0][0])
        df_activations = self._convert_to_df(activations_by_class)
        df_logits = self._convert_to_logits_df(logits_by_class)

        return df_activations, feature_count, df_logits

    def _convert_to_df(self, activations_by_class):
        # add class labels
        list(map(lambda f: f.insert(0, 0), activations_by_class[0]))
        list(map(lambda f: f.insert(0, 1), activations_by_class[1]))

        columns = ['Class'] + [f'feature_{idx}' for idx in range(len(activations_by_class[0][0][1:]))]
        df = pd.DataFrame(activations_by_class[0] + activations_by_class[1], columns=columns)

        return df

    def calculate_with_identifier(self):
        # shape: class: [[act1, act..., filename], [...]]
        activations_by_class, logits_by_class = return_feature_contribution_data_with_identifier(self.data_iterable, self.cnn, num_classes=2)

        if activations_by_class[0]:
            feature_count = len(activations_by_class[0][0]) - 1  # substract filename
        else:
            feature_count = len(activations_by_class[1][0]) - 1  # substract filename

        df_activations = self._convert_to_df_with_identifiers(activations_by_class)
        df_logits = self._convert_to_logits_df_with_identifiers(logits_by_class)

        return df_activations, feature_count, df_logits

    def _convert_to_df_with_identifiers(self, activations_by_class):
        # add class labels
        list(map(lambda f: f.insert(0, 0), activations_by_class[0]))
        list(map(lambda f: f.insert(0, 1), activations_by_class[1]))

        try:
            feature_range = len(activations_by_class[0][0][1:-1])
        except:
            feature_range = len(activations_by_class[1][0][1:-1])
        columns = ['Class'] + [f'feature_{idx}' for idx in range(feature_range)] + ['Filename']
        df = pd.DataFrame(activations_by_class[0] + activations_by_class[1], columns=columns)

        return df

    def _convert_to_logits_df_with_identifiers(self, logits_by_class):
        list(map(lambda f: f.insert(0, 0), logits_by_class[0]))
        list(map(lambda f: f.insert(0, 1), logits_by_class[1]))

        columns = ['Class'] + ['Logit_0', 'Logit_1'] + ['Filename']
        df = pd.DataFrame(logits_by_class[0] + logits_by_class[1], columns=columns)

        return df

    def _convert_to_logits_df(self, logits_by_class):
        list(map(lambda f: f.insert(0, 0), logits_by_class[0]))
        list(map(lambda f: f.insert(0, 1), logits_by_class[1]))

        columns = ['Class'] + ['Logit_0', 'Logit_1']
        df = pd.DataFrame(logits_by_class[0] + logits_by_class[1], columns=columns)

        return df

