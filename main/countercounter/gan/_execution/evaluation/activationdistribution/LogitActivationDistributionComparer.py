from main.countercounter.gan._execution.evaluation.activationdistribution.ActivationDistributionComparer import \
    ActivationDistributionComparer


class LogitActivationDistributionComparer(ActivationDistributionComparer):
    
    def __init__(self, path):
        super().__init__(path)
        self.name = 'logit'

    def _is_trivial(self, feature_triviality_by_feature_by_class_a, feature_triviality_by_feature_by_class_b,
                    feature, label):
        return False
