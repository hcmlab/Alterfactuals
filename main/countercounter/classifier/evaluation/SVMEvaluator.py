from os.path import sep

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from joblib import dump, load

from main.countercounter.classifier.evaluation.SVMDBDistanceCalculator import SVMDBDistanceCalculator
from main.countercounter.classifier.evaluation.SVMDBDistancePlotter import SVMDBDistancePlotter


class SVMEvaluator:

    def __init__(self, path_to_output_folder, path_to_activation_train_csv, path_to_activation_val_csv):
        self.path_to_output_folder = path_to_output_folder
        self.df_train = pd.read_csv(path_to_activation_train_csv, sep=';')
        self.df_val = pd.read_csv(path_to_activation_val_csv, sep=';')

    def evaluate(self):
        X_train, y_train = self._get_training_data(self.df_train)

        X_val, y_val = self._get_training_data(self.df_val)

        svm, search = self._train(X_train, y_train)
        self._eval(X_val, svm, y_val, search)
        self._store(svm)

        self._eval_db_distances(svm)

    def _eval_db_distances(self, svm):
        datasets = [
            self.df_train,
            self.df_val,
        ]

        labels = [
            'Train',
            'Val',
        ]

        name_dist_avg_std = SVMDBDistanceCalculator(svm, datasets, labels).calculate()

        SVMDBDistancePlotter(name_dist_avg_std, self.path_to_output_folder).plot()

    def _eval(self, X_val, svm, y_val, search):
        val_pred = svm.predict(X_val)

        accuracy = accuracy_score(y_val, val_pred)
        precision = precision_score(y_val, val_pred)
        recall = recall_score(y_val, val_pred)
        f1 = f1_score(y_val, val_pred)

        lines = [
            f'Best parameters: {search.best_params_}',
            f'Accuracy: {accuracy}\n',
            f'Precision: {precision}\n',
            f'Recall: {recall}\n',
            f'F1: {f1}\n',
        ]

        with open(f'{self.path_to_output_folder}{sep}svm_results.txt', 'w') as file:
            file.writelines(lines)

    def _get_training_data(self, df):
        X_columns = sorted([col for col in df if col.startswith('feature')])
        y_column = 'Class'

        return df[X_columns], df[y_column]

    def _train(self, X_train, y_train):
        param_grid = [
            {
                'C': [0.1, 1, 10, 100, 1000],
                'kernel': ['linear'],
                'max_iter': [500, 1000, 5000]
            },
        ]

        svc = SVC()

        search = GridSearchCV(
            estimator=svc,
            param_grid=param_grid,
            scoring='accuracy',
            cv=None,  # None == 5-fold cross validation
        )

        search.fit(X_train, y_train)

        svm = search.best_estimator_

        return svm, search

    def _store(self, svm):
        dump(svm, f'{self.path_to_output_folder}/svm.joblib')