
Preprocess:
    - Open the Project.ipynb notebook. 
    - Change the directory in PATH according to the project's location. 
    - Run all the code under subtitles 'Initialization' and 'Functions'.
    - If you want to train the model, and it is the first time you running it, you need to download the data.
      To do that, uncomment the code under 'Initialization > Downloading Data'
    

Train models: 
    - Under the subtitle of the model variation you ask to train (original model, Additional Features model and 
      tagging-based booster) run the cell load data.
    - Under 'Train Model', Change the hyper-parameters you want to check. You can also change the experiment name.
    - Run the code under the cell 'Train Model'.
    - To evaluate the trained model, change experiment_name to your model name
    - To display the experiments graphs, under cell 'Results', change experiment_name and run the cell 

Best-Model Evaluation:
    - Under the subtitle of the model variation you ask evaluate, run the cell load data.
    - Change experiment_name to "best_models/{dropout/emotions/poses/tagging_booster}"
    - Run the code under the cell 'Evaluation'.

