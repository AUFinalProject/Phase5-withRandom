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
