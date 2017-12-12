This repository contains a file with a sentiment classifier named 'classifier.py' which I used for my thesis. I used Logistic Regression (LogR) and a Support Vector Machine (SVM) to classify the tweets.

This repository also contains a file named 'data.py' which was used to preprocess all the data.

All the data was collected from the SemEval-2013 task 2 & the SemEval-2014 task 9. Please visit their website http://alt.qcri.org/semeval2014/task9/ for more information about the SemEval datasets.

The data consisted of 4 files:
- train_set.txt : this file contains 8.385 tweets that can be used to train the classifier
- dev_set.txt : this file contains 2.872 tweets that can be used to develop the classifier
- test_set.txt : this file contains 4.312 tweets that can be used to test the classifier
- sarcasm_test_set: this file contains 85 sarcastic tweets that can be used to test the classifier

In order to collect the tweets, you have to download the tweets yourself. Because it is illegal to publicly distribute the tweets. All lines in these files look like this:

ID1 <TAB> ID2 <TAB> label

Once you downloaded the tweets, all lines in the new downloaded file will look like this:

ID1 <TAB> ID <TAB> label <TAB> tweet

To download the tweets. Use the script download_tweets_api.py which is provided by SemEval-2013 & SemEval-2014 on their website. The commands you need to run the script are also provided. Please visit this website http://alt.qcri.org/semeval2014/task9/index.php?id=data-and-tools for this information.

The sarcastic tweets were collected from a corpus available on LWP workspace on the Univeristy of Groningen (RUG). Unfortunately, I could not distribute these tweets.
