from os.path import sep


class HurdleModelStatsPrinter:

    def __init__(self, stats, output_folder):
        self.stats = stats  # shape: (f'Feature {feature}', dist0, dist0_p_value, p_value0, dist1, dist1_p_value, p_value1)
        self.path = output_folder

    def print(self, type):
        feature_lines, critical_lines = self._print()

        lines = [
            'Calculated Hurdle Models with their p_values from Kolmogoriv-Smirnov:\n',
            '\n',
            f'{type} and classes where best model p_value <= 0.05:\n',
            f'Total: {len(critical_lines)}\n',
            '\n',
        ] + critical_lines + [
            '\n',
        ] + feature_lines

        with open(f'{self.path}{sep}hurdle_model_stats.txt', 'w') as file:
            file.writelines(lines)

    def _print(self):
        lines = []

        critical_values = []

        for stat in self.stats:
            feature, dist0, dist0_p_value, pvalues0, dist1, dist1_p_value, pvalues1 = stat

            lines.extend([
                '\n',
                f'{feature}\n',
                f'Class 0: all hurdle models: {pvalues0}\n',
                f'Class 0: chosen model: {dist0}\n',
                f'P-Value of chosen model: {dist0_p_value}\n'
                '\n',
                f'Class 1: all hurdle models: {pvalues1}\n',
                f'Class 1: chosen model: {dist1}\n',
                f'P-Value of chosen model: {dist1_p_value}\n'
                '\n',
                '-------------------'
            ])

            if dist0_p_value <= 0.05 and dist0_p_value != -1:
                critical_values.append(
                    f'{feature} class 0: {dist0_p_value}\n'
                )

            if dist1_p_value <= 0.05 and dist1_p_value != -1:
                critical_values.append(
                    f'{feature} class 1: {dist1_p_value}\n'
                )

        return lines, critical_values