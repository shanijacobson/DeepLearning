
Preprocess
    - Open EX2.ipynb notebook.
    - Change the directory in PATH according to the place you put this folder.
    - Run all the code under sub title ‘Initialization’ (Imports, Load Data and Network Architecture)

Train models:
    - First run the first code block under "Train Models".
    - Next select the modes you want to train and update the list “modes_to_train” as explained on the notebook and file_name.
    - Run this cell.
    - The Trained models are saved under "./Models/%" where % is the name of the selected mode. We saved both best model (on the validation set) and the final model.
    - The graphs are saved under "./Results/%" where % is the name of the selected mode. The name of the file is “file_name.png”

Evaluate performance of existing models:
    - Under "Train Models" run the first block.
    - In order to evaluate the results of the different existing models on the train and test dataset, run the code block under “Evaluation”.
