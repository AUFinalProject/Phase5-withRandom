# -------------------------------------------------------------------------------------------------------------------------------------------------------------
# Main, three machines (image - KMC, text - Logistic Regression, features - RF) and all white pdfs files
# @Authors:  Alexey Titov and Shir Bentabou
# @Version: 1.0
# @Date 08.2019
# -------------------------------------------------------------------------------------------------------------------------------------------------------------

# libraries
from classes.dataPDF import dataPDF
from classes.createDATA import createDATA
from classes.readPDF import readPDF
import os
import sys
import csv
import argparse
import tempfile
import numpy as np
from numpy import random
from array import *
# machine learning libraries
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import metrics
from sklearn.pipeline import Pipeline, make_pipeline
# importing K-Means
from sklearn.cluster import KMeans
# import RF
from sklearn.ensemble import RandomForestClassifier
# import LR
from sklearn.linear_model import LogisticRegression
# import AdaBoostClassifier
from sklearn.ensemble import AdaBoostClassifier
# import AdaBoostRegressor
from sklearn.ensemble import AdaBoostRegressor
# import XGBClassifier
from xgboost import XGBClassifier
# import XGBRegressor
from xgboost.sklearn import XGBRegressor
# import RFClassifier
from sklearn.ensemble import RandomForestClassifier
# import RFRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

if __name__ == "__main__":
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True,
                    help="path to input dataset")
    # arguments for k-means-clustering
    ap.add_argument("-c", "--clusters", type = int, default = 16,
		help="the number of clusters to form as well as the number of centroids to generate")
    ap.add_argument("-j", "--jobs", type = int, default = -1,
		help="the number of jobs to use for the computation. ")
    args = vars(ap.parse_args())
    # define the name of the directory to be created
    path_IMAGES = "IMAGES"
    path_TEXTS = "TEXTS"

    # create folders for images and texts
    try:
        os.mkdir(path_IMAGES)
        os.mkdir(path_TEXTS)
    except OSError:
        print("[!] Creation of the directories {} or {} failed, maybe the folders are exist".format(path_IMAGES, path_TEXTS))
    else:
        print(
            "[*] Successfully created the directories {} and {} ".format(path_IMAGES, path_TEXTS))
    folder_path = os.getcwd()
    dataset_path = os.path.join(folder_path, args["dataset"])

    # check if a folder of data is exist
    if (not os.path.exists(dataset_path)):
        print("[!] The {} folder is not exist!\n    GOODBYE".format(dataset_path))
        sys.exit()
 
    # create csv file
    with open("pdfFILES.csv", 'w') as csvFile:
            fields = ['File', 'Text']
            writer = csv.DictWriter(csvFile, fieldnames = fields)
            writer.writeheader()
    # start create data
    print("+++++++++++++++++++++++++++++++++++ START CREATE DATA +++++++++++++++++++++++++++++++++++")
    obj_data = createDATA(folder_path, args["dataset"])

    # convert first page of pdf file to image
    result = obj_data.convert(dataset_path)
    if (result):
        print("[*] Succces convert pdf files")
    else:
        print("[!] Whoops. something wrong dude. enable err var to track it")
        sys.exit()

    # extract JavaScript from pdf file
    result = obj_data.extract(dataset_path)
    if (result):
        print("[*] Succces extract JavaScript from pdf files")
    else:
        print("[!] Whoops. something wrong dude. enable err var to track it")
        sys.exit()
    print("\n+++++++++++++++++++++++++++++++++++++++++ FINISH ++++++++++++++++++++++++++++++++++++++++\n")
    
    # start create vectors
    print("++++++++++++++++++++++++++++++++++ START CREATE VECTORS +++++++++++++++++++++++++++++++++")
    # dir of folder and filter for pdf files
    files = [f for f in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, f))]
    files = list(filter(lambda f: f.endswith(('.pdf', '.PDF')), files))

    # variables for print information
    cnt_files = len(files)
    obj_pdfs_mal = []
    obj_pdfs_white = []
    labels_mal = []
    labels_white = []
    obj_read = readPDF(obj_data.getDict())
    # loop over the input pdfs
    for (i, pdfFILE) in enumerate(files):
        label = -1
        if ("mal" == pdfFILE.split(".")[0]):
           label = 1
           labels_mal.append(label)
           # create pdf object
           obj_pdf = dataPDF(pdfFILE, folder_path+'/', args["dataset"])
           obj_pdf.calculate_histogram_blur()
           obj_pdf.calculate_dsurlsjsentropy()
           obj_pdf.save_text(obj_read.extractTEXT(obj_pdf.getFilename(), obj_pdf.getImage()))
           obj_pdfs_mal.append(obj_pdf)
        else:
           label = 0
           labels_white.append(label)
           # create pdf object
           obj_pdf = dataPDF(pdfFILE, folder_path+'/', args["dataset"])
           obj_pdf.calculate_histogram_blur()
           obj_pdf.calculate_dsurlsjsentropy()
           obj_pdf.save_text(obj_read.extractTEXT(obj_pdf.getFilename(), obj_pdf.getImage()))
           obj_pdfs_white.append(obj_pdf)
        
        # show an update every 100 pdfs
        if (i > 0 and i % 100 == 0):
            print("[INFO] processed {}/{}".format(i, cnt_files))
    print("[INFO] processed {}/{}".format(cnt_files, cnt_files))
    print("\n+++++++++++++++++++++++++++++++++++++++++ FINISH ++++++++++++++++++++++++++++++++++++++++\n")
 
    # start machine learning
    print("+++++++++++++++++++++++++++++++++ START MACHINE LEARNING ++++++++++++++++++++++++++++++++")
    labels_mal = np.array(labels_mal)
    labels_white = np.array(labels_white)
    # partition the data into training and testing splits, using 70%
    # of the data for training and the remaining 30% for testing
    (trainF_mal, testF_mal, trainLabels_mal, testLabels_mal) = train_test_split(obj_pdfs_mal, labels_mal, test_size = 0.30, random_state = 42)
    (trainF_white, testF_white, trainLabels_white, testLabels_white) = train_test_split(obj_pdfs_white, labels_white, test_size = 0.30, random_state = 42)

    # partition the data into training and testing splits, using 67% (20% from all set)
    # of the data for training and the remaining 33% (10% from all set) for testing
    (testF_mal, valid_mal, testLabels_mal, validLabels_mal) = train_test_split(testF_mal, testLabels_mal, test_size = 0.33, random_state = 42)
    (testF_white, valid_white, testLabels_white, validLabels_white) = train_test_split(testF_white, testLabels_white, test_size = 0.33, random_state = 42)

    trainFeat = []
    testFeat = []
    validFeat = []
    trainLabels = []
    testLabels = []
    validLabels = []
    # malicious
    for pdf in trainF_mal:
        trainFeat.append(pdf.getImgHistogram())
    for label in trainLabels_mal:
        trainLabels.append(label)
    for pdf in testF_mal:
        testFeat.append(pdf.getImgHistogram())
    for label in testLabels_mal:
        testLabels.append(label)
    for pdf in valid_mal:
        validFeat.append(pdf.getImgHistogram())
    for label in validLabels_mal:
        validLabels.append(label)

    # benign
    for pdf in trainF_white:
        trainFeat.append(pdf.getImgHistogram())
    for label in trainLabels_white:
        trainLabels.append(label)
    for pdf in testF_white:
        testFeat.append(pdf.getImgHistogram())
    for label in testLabels_white:
        testLabels.append(label)
    for pdf in valid_white:
        validFeat.append(pdf.getImgHistogram())
    for label in validLabels_white:
        validLabels.append(label)    


    trainFeat = np.array(trainFeat)
    testFeat = np.array(testFeat)
    trainLabels = np.array(trainLabels)
    testLabels = np.array(testLabels)
    validFeat = np.array(validFeat)
    validLabels = np.array(validLabels)
    # instantiating kmeans
    km = KMeans(algorithm = 'auto', copy_x = True, init = 'k-means++', max_iter = 300, n_clusters = args["clusters"], n_init = 10, n_jobs = args["jobs"])

    # training km model
    km.fit(trainFeat)
    # testing km
    predictions1_m = km.predict(testFeat)
    valid1 = km.predict(validFeat)

    # creating vector for Random Forest on features
    trainFeat = []
    testFeat = []
    validFeat = []
   
    # malicious
    for pdf in trainF_mal:
        trainFeat.append(pdf.getFeatVec())
    for pdf in testF_mal:
        testFeat.append(pdf.getFeatVec())
    for pdf in valid_mal:
        validFeat.append(pdf.getFeatVec())

    # benign
    for pdf in trainF_white:
        trainFeat.append(pdf.getFeatVec())
    for pdf in testF_white:
        testFeat.append(pdf.getFeatVec())
    for pdf in valid_white:
        validFeat.append(pdf.getFeatVec())
   
    trainFeat = np.array(trainFeat)
    testFeat = np.array(testFeat)
    validFeat = np.array(validFeat)

    # instantiating Random Forest
    ranfor = Pipeline([
        ('clf', RandomForestClassifier(n_estimators = 30, random_state = 0)),
    ])
    ranfor.fit(trainFeat, trainLabels)
    predictions3 = ranfor.predict(testFeat)
    valid3 = ranfor.predict(validFeat)
    # creating vector for Logistic Regression on text
    trainFeat = []
    testFeat = []
    validFeat = []

    # malicious
    for pdf in trainF_mal:
        trainFeat.append(pdf.getText())
    for pdf in testF_mal:
        testFeat.append(pdf.getText())
    for pdf in valid_mal:
        validFeat.append(pdf.getText())

    # benign
    for pdf in trainF_white:
        trainFeat.append(pdf.getText())
    for pdf in testF_white:
        testFeat.append(pdf.getText())
    for pdf in valid_white:
        validFeat.append(pdf.getText())

    trainFeat = np.array(trainFeat)
    testFeat = np.array(testFeat)
    validFeat = np.array(validFeat)

    # instantiating Logistic Regression Machine
    logreg = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=1000, n_jobs=1, C=1e5)),
               ])
    logreg.fit(trainFeat, trainLabels)
    predictions2 = logreg.predict(testFeat)
    valid2 = logreg.predict(validFeat)
    print("\n+++++++++++++++++++++++++++++++++++++++++ FINISH ++++++++++++++++++++++++++++++++++++++++\n")

    # start boost
    print("+++++++++++++++++++++++++++++++++++++++ START BOOST +++++++++++++++++++++++++++++++++++++")
    # creating vectors
    trainFeat = []
    validFeat = []
    # train
    for p1, p2, p3 in zip(predictions1_m, predictions2, predictions3):
        p_all = [p1, p2, p3]
        trainFeat.append(p_all)
    # valid
    for p1, p2, p3 in zip(valid1, valid2, valid3):
        p_all = [p1, p2, p3]
        validFeat.append(p_all)

    trainFeat = np.array(trainFeat)
    validFeat = np.array(validFeat)

    # instantiating AdaBoostClassifier
    abc = AdaBoostClassifier(n_estimators = 100, random_state = 0)
    abc.fit(trainFeat, testLabels)
    print("Feature importances for AdaBoostClassifier: ")
    print(abc.feature_importances_)
    # make predictions for test data
    predictions = abc.predict(validFeat)
    accuracy = accuracy_score(validLabels, predictions)
    print("Accuracy of AdaBoostClassifier: %.2f%%" % (accuracy * 100.0))
    # classification_report - precision, recall, f1 table for adaboost classifier
    print(classification_report(validLabels, predictions, target_names=["benign", "malicious"]))
    cm = confusion_matrix(validLabels, predictions)
    # the count of true negatives is A00, false negatives is A10, true positives is A11 and false positives is A01
    print('confusion matrix:\n %s' % cm)

    # instantiating AdaBoostRegressor (similar to logistic regression)
    abr = AdaBoostRegressor(random_state = 0, n_estimators = 100)
    abr.fit(trainFeat, testLabels)
    print("Feature importances for AdaBoostRegressor: ")
    print(abr.feature_importances_)
    # make predictions for test data
    predictions = abr.predict(validFeat)
    accuracy = accuracy_score(validLabels, predictions.round())
    print("Accuracy of AdaBoostRegressor: %.2f%%" % (accuracy * 100.0))
    # classification_report - precision, recall, f1 table for adaboost classifier
    print(classification_report(validLabels, predictions.round(), target_names=["benign", "malicious"]))
    cm = confusion_matrix(validLabels, predictions.round())
    # the count of true negatives is A00, false negatives is A10, true positives is A11 and false positives is A01
    print('confusion matrix:\n %s' % cm)

    # instantiating XGBClassifier
    xgbc = XGBClassifier()
    xgbc.fit(trainFeat, testLabels)
    print("Feature importances for XGBClassifier: ")
    print(xgbc.feature_importances_)
    # make predictions for test data
    predictions = xgbc.predict(validFeat)
    accuracy = accuracy_score(validLabels, predictions)
    print("Accuracy of XGBClassifier: %.2f%%" % (accuracy * 100.0))
    # classification_report - precision, recall, f1 table for adaboost classifier
    print(classification_report(validLabels, predictions, target_names=["benign", "malicious"]))
    cm = confusion_matrix(validLabels, predictions)
    # the count of true negatives is A00, false negatives is A10, true positives is A11 and false positives is A01
    print('confusion matrix:\n %s' % cm)

    # instantiating XGBRegressor (similar to linear regression)
    xgbr = XGBRegressor(n_estimators = 100, max_depth = 3)
    xgbr.fit(trainFeat, testLabels)
    print("Feature importances for XGBRegressor: ")
    print(xgbr.feature_importances_)
    # make predictions for test data
    predictions = xgbr.predict(validFeat)
    accuracy = accuracy_score(validLabels, predictions.round())
    print("Accuracy of XGBRegressor: %.2f%%" % (accuracy * 100.0))
    # classification_report - precision, recall, f1 table for adaboost classifier
    print(classification_report(validLabels, predictions.round(), target_names=["benign", "malicious"]))
    cm = confusion_matrix(validLabels, predictions.round())
    # the count of true negatives is A00, false negatives is A10, true positives is A11 and false positives is A01
    print('confusion matrix:\n %s' % cm)

    # instantiating Random Forest Classifier
    rfclf = RandomForestClassifier(n_estimators = 250)
    rfclf.fit(trainFeat, testLabels)
    print("Feature importances for Random Forest Classifier: ")
    print(rfclf.feature_importances_)
    # predictions for test data
    cla_pred = rfclf.predict(validFeat)
    rf_acc = accuracy_score(validLabels, cla_pred)
    print("Random Forest Accuracy: %.2f%%" % (rf_acc * 100.0))
    # classification_report - precision, recall, f1 table for adaboost classifier
    print(classification_report(validLabels, cla_pred, target_names=["benign", "malicious"]))
    # confusion_matrix
    cm_rf_cla = confusion_matrix(validLabels, cla_pred)
    # the count of true negatives is A00, false negatives is A10, true positives is A11 and false positives is A01
    print('confusion matrix:\n %s' % cm_rf_cla)
    
    # instantiating Random Forest Regressor
    rfreg = RandomForestRegressor(n_estimators = 250)
    rfreg.fit(trainFeat, testLabels)
    print("Feature importances for Random Forest Regressor: ")
    print(rfreg.feature_importances_)
    # predictions for test data
    reg_pred = rfreg.predict(validFeat)
    rfreg_acc = accuracy_score(validLabels, reg_pred.round())
    print("Random Forest Accuracy: %.2f%%" % (rfreg_acc * 100.0))
    # classification_report - precision, recall, f1 table for adaboost classifier
    print(classification_report(validLabels, reg_pred.round(), target_names=["benign", "malicious"]))
    # confusion_matrix
    cm_rf_reg = confusion_matrix(validLabels, reg_pred.round())
    # the count of true negatives is A00, false negatives is A10, true positives is A11 and false positives is A01
    print('confusion matrix:\n %s' % cm_rf_reg)
    print("\n+++++++++++++++++++++++++++++++++++++++++ FINISH ++++++++++++++++++++++++++++++++++++++++\n")
