from main.countercounter.autoencoder.evaluation.EvaluationPipeline import EvaluationPipeline

config_nr = '002'
root_dir = 'TODO'
config_dir = 'EmClass/configs'

for epoch in range(12, 25):
    EvaluationPipeline().run(config_nr, root_dir, config_dir, f'epoch_{epoch}')
    print('-----------------')