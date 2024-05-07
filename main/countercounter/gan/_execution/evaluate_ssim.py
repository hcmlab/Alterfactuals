import os

from main.countercounter.gan._execution.evaluation.metrics.proximity.SSIMEvaluator import SSIMEvaluator

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


root_dir = 'TODO'
config_dir = 'EmClass/GANConfigs'


if __name__ == '__main__':

    confs = [

        ('132', 74, 'FashionMNIST', 'TODO', None),
    ]
    for (run_nr, epoch, type, ssim_csv, ssim_key) in confs:
        path = 'TODO'

        SSIMEvaluator(path, ssim_csv, ssim_key).evaluate()

