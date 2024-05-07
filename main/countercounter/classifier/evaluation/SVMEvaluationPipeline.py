from main.countercounter.classifier.evaluation.SVMEvaluator import SVMEvaluator
from main.countercounter.classifier.training.FilePreparation import FilePreparation
from main.countercounter.classifier.training.Setup import Setup


class SVMEvaluationPipeline:

    def run(self, config_nr, checkpoint, path_to_output_folder=None, path_to_activation_train_csv=None, path_to_activation_val_csv=None):
        path_to_output_folder = FilePreparation(Setup()).make_output_dir_for_run(path_to_output_folder, config_nr, checkpoint)
        SVMEvaluator(path_to_output_folder, path_to_activation_train_csv, path_to_activation_val_csv).evaluate()