Preprocess:
    - Open Q3.4.ipynb notebook.
    - Change the directory in PATH according to the place you put this folder.
    - Run all the code under sub title ‘Initialization’ (Imports, Load Data and Network Architecture) 

Train models:
    - Run the first code blocks under "Train Models".
    - Run the code block  of the model you want to train (DCGAN or WGAN)
    - The Trained models are saved under "./Models/{model_name}/{file_name}.pt", where model_name is DCGAN/WGAN.
    - The graphs are saved under "./Results/{model_name}/{file_name}.png"  

Evaluate performance of existing models 
    - Under "Evaluation" run the first block.
    - Run the block code of the model you want to evaluate (DCGAN or WGAN)
