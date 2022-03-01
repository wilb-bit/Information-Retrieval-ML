code requires the following files for execution:  
1.trec_eval.exe (if using Windows, might differ for a different operating system)  
2.train.qrels

the following files will be created due to use of trec_eval:  
1.current.run (output_file)  
2.fold.tsv (run_file)  

variable 'sweep' is used to specify whether parameter sweep will run.  
need to open and change this variable manually.  
if sweep is TRUE - will set paramaters  
 if sweep if FALSE - will run kfold sweep.  
