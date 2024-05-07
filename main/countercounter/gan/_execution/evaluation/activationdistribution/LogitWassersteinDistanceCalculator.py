from scipy.stats import wasserstein_distance

from main.countercounter.gan._execution.evaluation.activationdistribution.WassersteinDistanceCalculator import \
    WassersteinDistanceCalculator


class LogitWassersteinDistanceCalculator(WassersteinDistanceCalculator):

    def __init__(self):
        super().__init__()

    def _is_trivial(self, feature, feature_triviality_by_feature_by_class_gen,
                    feature_triviality_by_feature_by_class_test, label):
        return False
