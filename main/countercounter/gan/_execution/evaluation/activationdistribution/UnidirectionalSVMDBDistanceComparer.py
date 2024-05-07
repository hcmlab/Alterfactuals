import numpy as np
from statsmodels.robust.scale import qn_scale


class UnidirectionalSVMDBDistanceComparer:
    
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

            raw_distances = np.array(test_dist) - np.array(gen_dist)

            # if distance has increased, set distance to 0
            filtered_distances = np.maximum(raw_distances, 0)

            # caution: this can produce nan if scale_factor is 0
            # filtered_distances = np.abs(filtered_distances / scale_factor)

            distance_differences.append((f'{test_name}-{gen_name}', filtered_distances))

        return distance_differences

    def _get_scale_factors(self):
        scale_factors = []

        for _, dist, _, _ in self.train_name_dist_avg_std:
            scale_factors.append(qn_scale(dist))

        return scale_factors

    def compare_distance_sign(self):
        count_of_same_sign_before_and_after = {}

        for idx, (gen_name, gen_dist, _, _) in enumerate(self.gen_name_dist_avg_std):
            test_name, test_dist, _, _ = self.test_name_dist_avg_std[idx]

            if not isinstance(gen_dist, np.ndarray):
                gen_dist = np.array(gen_dist)
            if not isinstance(test_dist, np.ndarray):
                test_dist = np.array(test_dist)

            # check if sign of distance has switched (or at least: not stayed the same)
            gen_dist_sign = np.sign(gen_dist)
            test_dist_sign = np.sign(test_dist)

            sign_combined = gen_dist_sign + test_dist_sign
            same_sign_before_after = np.argwhere(np.abs(sign_combined) == 2)  # signs were either -1 and -1 or 1 and 1 --> same sign
            count_of_same_sign_before_and_after[f'{test_name}--{gen_name}'] = (len(same_sign_before_after), len(sign_combined))

        return count_of_same_sign_before_and_after
