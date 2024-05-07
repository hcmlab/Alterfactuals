This is the code for our IJCAI 2024 paper: 
'Relevant Irrelevance: Generating Alterfactual Explanations for Image Classifiers'

Install all required libraries listed in requirements.txt


Depending on what you want to train, navigate to the corresponding directory. E.g. if you want to train a classifier, head to /countercounter/classifier.
To traina GAN, set your config in countercounter/gan/_execution/configs and run /countercounter/gan/_execution/execute.

To train another type of model: The folder structure is pretty much the same for all models. Some have an explicity training.py file. If this exists, use this to train.
