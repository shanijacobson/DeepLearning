
Preprocess:
    - Open Q3.3.ipynb notebook.
    - Change the directory in PATH according to the place you put this folder.
    - Run all the code under sub title ‘Initialization’ (Imports, Load Data and Network Architecture) 

Train auto-encoder model:
    - Run the code blocks under "Train Autoencoder Model".
    - The Trained models are saved under "./Models/{file_name}.pt".
    - The graphs are saved under "./Results/{file_name}.png"  

Train svm model:
    - Under "Train SVM Model", change  "auto_encoder_file_name" according to the file name of the autoencoder you want to train on.
    - Run the code blocks under "Train SVM Model".
    - The Trained models are saved under "./Models/{file_name}_{bm}.pt".

Evaluate performance of existing models: 
    - Under "Train SVM Models" run the first block.
    - In order to evaluate the results of the different existing models, run the code block under “Evaluation”.

