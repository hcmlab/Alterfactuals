from main.countercounter.classifier.evaluation.DistributionPlotter import DistributionPlotter


class LogitDistributionPlotter(DistributionPlotter):

    def __init__(self, path_to_output_folder):
        super().__init__(path_to_output_folder)
        self.name = 'Logit'
        self.type = ''
