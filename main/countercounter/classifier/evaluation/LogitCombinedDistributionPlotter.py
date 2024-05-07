from main.countercounter.classifier.evaluation.CombinedDistributionPlotter import CombinedDistributionPlotter


class LogitCombinedDistributionPlotter(CombinedDistributionPlotter):

    def __init__(self, path_to_output_folder):
        super().__init__(path_to_output_folder)
        self.name = 'Logit'
