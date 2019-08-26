# Phase5-withRandom
The fifth phase of our project. In this phase we are hoping to create a final classifier of PDF files.
The final classifier will be based on the three previous machines in our project: image, text, and features.
The process of the final machine will be the following:
  * install: `sudo pip3 install xgboost`
  * Extract all data needed for the three base machines (image, text, features) - this is done using classes, imported into as.py.
  * Create base vectors for every sample (image, text, features)
  * Run every base machine on the samples, and return the calssification of the sample by every machine.
  * Create a vector for the boost algorithm from the base machines classifications for every sample.
  * Run boost algorithm with RF on sample boost vectors.
  * Return boost algorithm accuracy.
  
all_random.png:
 * AdaBoostClassifier: 8171 - true, 568 - false, accuracy -  93.5%.
 * AdaBoostRegressor: 8414 - true, 325 - false, accuracy -  96.28%.
 * XGBClassifier: 8180 - true, 559- false, accuracy -  93.6%.
 * XGBRegressor: 8216 - true, 523 - false, accuracy -  94.01%.
 * Random Forest Classifier: 7868 - true, 871 - false, accuracy -  90.03%.
 * Random Forest Regressor: 8216 - true, 523 - false, accuracy -  94.01%.
