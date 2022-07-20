
Preprocess:
    - Open the MINST_CNN.ipynb notebook.
    - Change the directory in PATH according to the place you saved the existing models.
    - Run all the code under sub title ‘Initialization’ (Imports, Load Data and Network Architecture)

Train models:
    - First run the first code block under "Train Models".
    - Next run the code according to the wanted techniques (i.e. Original, Dropout, Weight Decay, or Bath Normalization)
    - It is possible to run all the blocks under "Train Models".
    - The Trained models are saved under "./Models/%" where % is the name of the selected mode.
    - The graphs are saved under "./ex1_313581803_314882861/Results/%" with the relevant hyperparameters as the name of the graph. Evaluate performance of existing models
    - Under "Train Models" run the first block.
    - In order to evaluate the results of the different existing models on the train and test dataset, run the code block under “Evaluation”.

Finding Hyperparameters
    - If you wish to find the best hyperparameters to the model, with train and validation sets, run the "Finding Hyperparameters" Code block.
    - To change learning rates been tested, change the following line: o lr_list = np.arange(0.0005, 0.001, 0.00005)
    - To change the batch sizes been tested, change the following line: o batch_size = [128,256, 64]
    - The Code will generate plots that can be found in "./DeepLearing/Assignment1/Results/Original"
    - The name of the plot represents the hyperparameters used