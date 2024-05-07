from os.path import sep


class CriticalFeatureMismatchPlotter:

    def __init__(self, critical_feature_mismatches, output_folder):
        self.mismatches = critical_feature_mismatches

        self.output_folder = output_folder

    def plot(self):
        lines = [
            f'For the following features, no distances could be calculated with the hurdle models.\n',
            f'This is not generally a problem, e.g. if it is a trivial feature (i.e. values always 0).\n',
            f'However, in the following features, there the value of test and generated were not the same.\n',
            '\n',
            f'Compare this with the MAD and Qn approach.\n',
            '\n',
            '\n',
        ]

        for feature, label, test_feature_value, gen_feature_value in self.mismatches:
           lines.append(f'Different values (test: {test_feature_value} - gen: {gen_feature_value} found for feature {feature} in class {label})\n')

        with open(f'{self.output_folder}{sep}critical_feature_distance_mismatches.txt', 'w') as file:
            file.writelines(lines)