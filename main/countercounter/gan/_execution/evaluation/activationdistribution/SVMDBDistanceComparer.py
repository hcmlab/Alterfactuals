import numpy as np
from statsmodels.robust.scale import qn_scale


class SVMDBDistanceComparer:
    
    def __init__(self, train_name_dist_avg_std, test_name_dist_avg_std, gen_name_dist_avg_std, output_folder):
        self.train_name_dist_avg_std = train_name_dist_avg_std
        self.test_name_dist_avg_std = test_name_dist_avg_std
        self.gen_name_dist_avg_std = gen_name_dist_avg_std
        
        self.output_folder = output_folder

    def compare(self):
        # scale_factors = self._get_scale_factors()
        
        assert len(self.train_name_dist_avg_std) == len(self.test_name_dist_avg_std) == len(self.gen_name_dist_avg_std)

        distance_differences = []

        for idx, (gen_name, gen_dist, _, _) in enumerate(self.gen_name_dist_avg_std):
            test_name, test_dist, _, _ = self.test_name_dist_avg_std[idx]
            
            # scale_factor = scale_factors[idx]

            # caution: this can produce nan if scale_factor is 0
            distances = np.abs((np.array(gen_dist) - np.array(test_dist))) #/ scale_factor

            distance_differences.append((f'{test_name}-{gen_name}', distances))

        return distance_differences

    def _get_scale_factors(self):
        scale_factors = []

        for _, dist, _, _ in self.train_name_dist_avg_std:
            scale_factors.append(qn_scale(dist))

        return scale_factors
