from main.countercounter.gan._execution.evaluation.activationdistribution.DistributionDistancePlotter import \
    DistributionDistancePlotter


class LogitDistributionDistancePlotter(DistributionDistancePlotter):

    def __init__(self, distances_by_feature_by_class, total_distances, path_to_output_folder):
        super().__init__(distances_by_feature_by_class, total_distances, path_to_output_folder,
                         None)
        self.name = 'logit'
