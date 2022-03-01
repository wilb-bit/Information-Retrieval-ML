# Information retrival system using query document pairs and their features

train.tsv contains these pairs and their features and their relevence label (0 being irrelevent, 1 being parially relevent and 2 being relevent).    
test.tsv just contains the same however without the label.  
As this is a univeristy assigment we must use train.tsv to train our model and then predict on test.tsv which will then be marked.  

## LambdaRank
LambdaRank is used for this problem. LGBMRanker from the lightgbm library. A gridsearch is run to find the best hyperparameters.   
NDCG score is used to validate the efficiency of the model which is calculated using trec_eval.exe.   
train.qrels is used by trec_eval.exe and contains the query document pairs and their label that are in train.tsv.

the following files will be created due to use of trec_eval:  
1.current.run (output_file)  
2.fold.tsv (run_file)  

## Running the program (A2.py)
variable 'sweep' is used to specify whether parameter sweep will run.  
need to open and change this variable manually.  
if sweep is TRUE - will set paramaters  
if sweep if FALSE - will run kfold sweep.  

trec_eval.exe is also selected as the code was run in windows. if running in Mac or linux the other trec_eval executable will need to be run. Edit the code specifying to use the appropriate trec_eval executable.
