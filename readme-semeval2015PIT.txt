

====================  PIT 2015  ====================
  SemEval-2015 Task 1:
  Paraphrase and Semantic Similarity in Twitter
====================================================
			    
ORGANIZERS

  Wei Xu, University of Pennsylvania
  Chris Callison-Burch, University of Pennsylvania
  Bill Dolan, Microsoft Research


TRAIN/DEV DATA

  The dataset contains the following files:
  
    ./data/readme.txt (this file)
    ./data/train.data (13063 sentence pairs)
    ./data/dev.data   (4727 sentence pairs)

  Notice that the train and dev data is collected from the same time period and
  same trending topics. In the evaluation later, we will test the system on data
  collected from a different time period.

  Both data files come in the tab-separated format. Each line contains 6 columns:
    
    | Trending_Topic_Name | Sent_1 | Sent_2 | Label | Sent_1_tag | Sent_2_tag |
 
  The "Trending_Topic_Name" are the names of trends provided by Twitter, which are
  not hashtags.
  
  The "Sent_1" and "Sent_2" are the two sentences, which are not necessarily full 
  tweets. Tweets were tokenized (thanks to Brendan O'Connor et al.) and 
  split into sentences. 
 
  The "Label" column is in a format such like "(1, 4)", which means among 5 votes 
  from Amazon Mechanical turkers only 1 is positive and 4 are negative. We would 
  suggest map them to binary labels as follows:
    
    paraphrases: (3, 2) (4, 1) (5, 0)
    non-paraphrases: (1, 4) (0, 5)
    debatable: (2, 3)  which you may discard if training binary classifier

  The "Sent_1_tag" and "Sent_2_tag" are the two sentences with part-of-speech 
  and named entity tags (thanks to Alan Ritter). 
  
        
EVALUATION

  The input will be in similar format as training/dev data. The participant will 
  be required to produce a binary label (paraphrase) for each sentence pair, and 
  optionally a real number between 0 ~ 1 for measuring semantic similarity.
  
  Evaluation metrics are not finally set yet. It likely to be Precision/Recall/F1
  for the paraphrase outputs, and Pearson correlation for evaluating those systems
  that output additional degreed scores. 
  
  We will release more details about evaluation in November 2014, and send out 
  announcements through our website and mailing list:
     http://alt.qcri.org/semeval2015/task1/
     https://groups.google.com/group/semeval-paraphrase 
  
  
BASELINE
  
  A logistic regression model using simple lexical overlap features:
    ./script/baseline_logisticregression.py

  It is our reimplementation in Python. This basline was originally 
  used by Dipanjan Das and Noah A. Smith in their ACL 2009 paper
  "Paraphrase Identification as Probabilistic Quasi-Synchronous Recognition".

  To run the script, you will need to install NLTK and Megam packages:
  http://www.nltk.org/_modules/nltk/classify/megam.html
  If you have troubles with Megam, you may need to rebuild it from source code:
  http://stackoverflow.com/questions/11071901/stuck-in-using-megam-in-python-nltk-classify-maxentclassifier

  Example output, if training on train.data and test on dev.data will look like:
    
    Read in 11513 training data ...  (after discarding the data with debatable cases)
    Read in 4139 test data ...       (see details in TRAIN/DEV DATA section)
    PRECISION: 0.704069050555
    RECALL:    0.389229720518
    F1:        0.501316944688
    ACCURACY:  0.725537569461 

  The script will provide the numbers for plotting precision/recall curves, or a 
  single 
  

REFERENCES 

   (details about how this data was collected and some analysis is in Chapter 6)
   Wei Xu (2014). Data-Drive Approaches for Paraphrasing Across Language Variations. 
   PhD thesis, Department of Computer Science, New York University.  
   http://www.cis.upenn.edu/~xwe/files/thesis-wei.pdf			    
			    