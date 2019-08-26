# -------------------------------------------------------------------------------------------------------------------------------------------------------------
# Main, three machines (image - KMC, text - Logistic Regression, features - RF) and all white pdfs files
# @Authors:  Alexey Titov and Shir Bentabou
# @Version: 1.0
# @Date 06.2019
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
    obj_pdfs = []
    labels = []
    obj_read = readPDF(obj_data.getDict())
    set_white_files = []
    # loop over the input pdfs
    for (i, pdfFILE) in enumerate(files):
        label = -1
        if ("mal" == pdfFILE.split(".")[0]):
           label = 1
        else:
           label = 0
           set_white_files.append(pdfFILE)
        labels.append(label)
        # create pdf object
        obj_pdf = dataPDF(pdfFILE, folder_path+'/', args["dataset"])
        obj_pdf.calculate_histogram_blur()
        obj_pdf.calculate_dsurlsjsentropy()
        obj_pdf.save_text(obj_read.extractTEXT(obj_pdf.getFilename(), obj_pdf.getImage()))
        obj_pdfs.append(obj_pdf)
        # show an update every 50 pdfs
        if (i > 0 and i % 50 == 0):
            print("[INFO] processed {}/{}".format(i, cnt_files))
    print("[INFO] processed {}/{}".format(cnt_files, cnt_files))
    print("\n+++++++++++++++++++++++++++++++++++++++++ FINISH ++++++++++++++++++++++++++++++++++++++++\n")
 
    # start machine learning
    print("+++++++++++++++++++++++++++++++++ START MACHINE LEARNING ++++++++++++++++++++++++++++++++")
    labels = np.array(labels)
    # partition the data into training and testing splits, using 70%
    # of the data for training and the remaining 30% for testing
    (trainF, testF, trainLabels, testLabels) = train_test_split(obj_pdfs, labels, test_size = 0.30, random_state = 42)
    trainFeat = []
    testFeat = []
    for pdf in trainF:
        trainFeat.append(pdf.getImgHistogram())
    for pdf in testF:
        testFeat.append(pdf.getImgHistogram())
    trainFeat = np.array(trainFeat)
    testFeat = np.array(testFeat)
    # instantiating kmeans
    km = KMeans(algorithm = 'auto', copy_x = True, init = 'k-means++', max_iter = 300, n_clusters = args["clusters"], n_init = 10, n_jobs = args["jobs"])

    # training km model
    km.fit(trainFeat)
    # testing km
    predictions1_m = km.predict(testFeat)

    # creating vector for Random Forest on features
    trainFeat = []
    testFeat = []
    for pdf in trainF:
        trainFeat.append(pdf.getFeatVec())
    for pdf in testF:
        testFeat.append(pdf.getFeatVec())
    trainFeat = np.array(trainFeat)
    testFeat = np.array(testFeat)
    
    # instantiating Random Forest
    ranfor = Pipeline([
        ('clf', RandomForestClassifier(n_estimators = 30, random_state = 0)),
    ])
    ranfor.fit(trainFeat, trainLabels)
    predictions3 = ranfor.predict(testFeat)

    # creating vector for Logistic Regression on text
    trainFeat = []
    testFeat = []
    for pdf in trainF:
        trainFeat.append(pdf.getText())
    for pdf in testF:
        testFeat.append(pdf.getText())
    # instantiating Logistic Regression Machine
    logreg = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=1000, n_jobs=1, C=1e5)),
               ])
    logreg.fit(trainFeat, trainLabels)
    predictions2 = logreg.predict(testFeat)
    print("\n+++++++++++++++++++++++++++++++++++++++++ FINISH ++++++++++++++++++++++++++++++++++++++++\n")

    # start boost
    print("+++++++++++++++++++++++++++++++++++++++ START BOOST +++++++++++++++++++++++++++++++++++++")
    # creating vectors
    trainFeat = []
    for p1, p2, p3 in zip(predictions1_m, predictions2, predictions3):
        p_all = [p1, p2, p3]
        trainFeat.append(p_all)
    trainFeat = np.array(trainFeat)
    # partition the data into training and testing splits, using 67% (20% from all set)
    # of the data for training and the remaining 33% (10% from all set) for testing
    (trainFeat, testFeat, trainLabels, testLabels) = train_test_split(trainFeat, testLabels, test_size = 0.33, random_state = 42)

    # instantiating AdaBoostClassifier
    abc = AdaBoostClassifier(n_estimators = 100, random_state = 0)
    abc.fit(trainFeat, trainLabels)
    print("Feature importances for AdaBoostClassifier: ")
    print(abc.feature_importances_)
    # make predictions for test data
    predictions = abc.predict(testFeat)
    accuracy = accuracy_score(testLabels, predictions)
    print("Accuracy of AdaBoostClassifier: %.2f%%" % (accuracy * 100.0))
    # classification_report - precision, recall, f1 table for adaboost classifier
    print(classification_report(testLabels, predictions, target_names=["benign", "malicious"]))
    cm = confusion_matrix(testLabels, predictions)
    # the count of true negatives is A00, false negatives is A10, true positives is A11 and false positives is A01
    print('confusion matrix:\n %s' % cm)

    # instantiating AdaBoostRegressor (similar to logistic regression)
    abr = AdaBoostRegressor(random_state = 0, n_estimators = 100)
    abr.fit(trainFeat, trainLabels)
    print("Feature importances for AdaBoostRegressor: ")
    print(abr.feature_importances_)
    # make predictions for test data
    predictions = abr.predict(testFeat)
    accuracy = accuracy_score(testLabels, predictions.round())
    print("Accuracy of AdaBoostRegressor: %.2f%%" % (accuracy * 100.0))
    # classification_report - precision, recall, f1 table for adaboost classifier
    print(classification_report(testLabels, predictions.round(), target_names=["benign", "malicious"]))
    cm = confusion_matrix(testLabels, predictions.round())
    # the count of true negatives is A00, false negatives is A10, true positives is A11 and false positives is A01
    print('confusion matrix:\n %s' % cm)

    # instantiating XGBClassifier
    xgbc = XGBClassifier()
    xgbc.fit(trainFeat, trainLabels)
    print("Feature importances for XGBClassifier: ")
    print(xgbc.feature_importances_)
    # make predictions for test data
    predictions = xgbc.predict(testFeat)
    accuracy = accuracy_score(testLabels, predictions)
    print("Accuracy of XGBClassifier: %.2f%%" % (accuracy * 100.0))
    # classification_report - precision, recall, f1 table for adaboost classifier
    print(classification_report(testLabels, predictions, target_names=["benign", "malicious"]))
    cm = confusion_matrix(testLabels, predictions)
    # the count of true negatives is A00, false negatives is A10, true positives is A11 and false positives is A01
    print('confusion matrix:\n %s' % cm)

    # instantiating XGBRegressor (similar to linear regression)
    xgbr = XGBRegressor(n_estimators = 100, max_depth = 3)
    xgbr.fit(trainFeat, trainLabels)
    print("Feature importances for XGBRegressor: ")
    print(xgbr.feature_importances_)
    # make predictions for test data
    predictions = xgbr.predict(testFeat)
    accuracy = accuracy_score(testLabels, predictions.round())
    print("Accuracy of XGBRegressor: %.2f%%" % (accuracy * 100.0))
    # classification_report - precision, recall, f1 table for adaboost classifier
    print(classification_report(testLabels, predictions.round(), target_names=["benign", "malicious"]))
    cm = confusion_matrix(testLabels, predictions.round())
    # the count of true negatives is A00, false negatives is A10, true positives is A11 and false positives is A01
    print('confusion matrix:\n %s' % cm)

    # instantiating Random Forest Classifier
    rfclf = RandomForestClassifier(n_estimators = 250)
    rfclf.fit(trainFeat, trainLabels)
    print("Feature importances for Random Forest: ")
    print(rfclf.feature_importances_)
    # predictions for test data
    cla_pred = rfclf.predict(testFeat)
    rf_acc = accuracy_score(testLabels, cla_pred)
    print("Random Forest Accuracy: %.2f%%" % (rf_acc * 100.0))
    # classification_report - precision, recall, f1 table for adaboost classifier
    print(classification_report(testLabels, cla_pred, target_names=["benign", "malicious"]))
    # confusion_matrix
    cm_rf_cla = confusion_matrix(testLabels, cla_pred)
    # the count of true negatives is A00, false negatives is A10, true positives is A11 and false positives is A01
    print('confusion matrix:\n %s' % cm_rf_cla)
    
    # instantiating Random Forest Regressor
    rfreg = RandomForestRegressor(n_estimators = 250)
    rfreg.fit(trainFeat, trainLabels)
    print("Feature importances for Random Forest: ")
    print(rfreg.feature_importances_)
    # predictions for test data
    reg_pred = rfreg.predict(testFeat)
    rfreg_acc = accuracy_score(testLabels, reg_pred.round())
    print("Random Forest Accuracy: %.2f%%" % (rfreg_acc * 100.0))
    # classification_report - precision, recall, f1 table for adaboost classifier
    print(classification_report(testLabels, reg_pred.round(), target_names=["benign", "malicious"]))
    # confusion_matrix
    cm_rf_reg = confusion_matrix(testLabels, reg_pred.round())
    # the count of true negatives is A00, false negatives is A10, true positives is A11 and false positives is A01
    print('confusion matrix:\n %s' % cm_rf_reg)
    print("\n+++++++++++++++++++++++++++++++++++++++++ FINISH ++++++++++++++++++++++++++++++++++++++++\n")
   
    # start check all white pdf files
    print("+++++++++++++++++++++++++++++++++++++++ START CHECK +++++++++++++++++++++++++++++++++++++")
    white_path = 'WHITE'
    dataset_path = os.path.join(folder_path, white_path)
    # extract JavaScript from white pdf file
    result = obj_data.extract(dataset_path)
    if (result):
        print("[*] Succces extract JavaScript from white pdf files")
    else:
        print("[!] Whoops. something wrong dude. enable err var to track it")
        sys.exit()

    # dir of folder and filter for pdf files
    files = [f for f in os.listdir(dataset_path) if os.path.isfile(
        os.path.join(dataset_path, f))]
    files = list(filter(lambda f: f.endswith(('.pdf', '.PDF')), files))

    #           AdoBC   AdoBR   XGBC    XGBR    RFC     RFR
    answers = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
    # loop over the input pdfs
    for (i, pdfFILE) in enumerate(files):
        if (pdfFILE in set_white_files):
            continue
        obj_data.convertOnePDF(dataset_path + '/' + pdfFILE)
        obj_read = readPDF(obj_data.getDict())
        # create pdf object
        obj_pdf = dataPDF(pdfFILE, folder_path+'/', white_path)
        
        # Image
        vector1 = []
        obj_pdf.calculate_histogram_blur()
        vector1.append(obj_pdf.getImgHistogram())
        vector1 = np.array(vector1)
        
        # 28 features
        vector2 = []
        obj_pdf.calculate_dsurlsjsentropy()
        vector2.append(obj_pdf.getFeatVec())
        vector2 = np.array(vector2)
    
        # Text
        obj_pdf.save_text(obj_read.extractTEXT(obj_pdf.getFilename(), obj_pdf.getImage()))
        vector_text = []
        vector_text.append(obj_pdf.getText())
        
        v1 = km.predict(vector1)
        v2 = logreg.predict(vector_text)
        v3 = ranfor.predict(vector2)
        v_all = [[v1[0], v2[0], v3[0]]]
  
        answer = abc.predict(v_all)
        if (answer == 0):
            answers[0][0] = answers[0][0] + 1
        else:
            answers[0][1] = answers[0][1] + 1
            print("AdoBoostClassifier ",pdfFILE)
        answer = abr.predict(v_all)
        if (answer.round() == 0):
            answers[1][0] = answers[1][0] + 1
        else:
            answers[1][1] = answers[1][1] + 1
            print("AdoBoostRegression ",pdfFILE)
        answer = xgbc.predict(v_all)
        if (answer == 0):
            answers[2][0] = answers[2][0] + 1
        else:
            answers[2][1] = answers[2][1] + 1
            print("XGBClassifier ",pdfFILE)
        answer = xgbr.predict(v_all)
        if (answer.round() == 0):
            answers[3][0] = answers[3][0] + 1
        else:
            answers[3][1] = answers[3][1] + 1
            print("XGBRegression ",pdfFILE)
        answer = rfclf.predict(v_all)
        if (answer == 0):
            answers[4][0] = answers[4][0] + 1
        else:
            answers[4][1] = answers[4][1] + 1
            print("RFClassifier ",pdfFILE)
        answer = rfreg.predict(v_all)
        if (answer.round() == 0):
            answers[5][0] = answers[5][0] + 1
        else:
            answers[5][1] = answers[5][1] + 1
            print("RFRegression ",pdfFILE)
        try:
            obj_pdf.removeIMAGE()
        except:
            continue
    print(answers)
    print("\n+++++++++++++++++++++++++++++++++++++++++ FINISH ++++++++++++++++++++++++++++++++++++++++\n")
