# -------------------------------------------------------------------------------------------------------------------------------------------------------------
# The our class for PDF files
# @Authors:  Alexey Titov and Shir Bentabou
# @Version: 1.0
# -------------------------------------------------------------------------------------------------------------------------------------------------------------

# libraries
import numpy as np
import subprocess
import imutils
import cv2
import re
import os
import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import csv


# global lists
port_good = [":443/", ":80/", ":8080/"]
bad_word = ["target", "&", "?", "download", "php", "loader", "login", "=", "+"]
default_features = ["obj", "endobj", "stream", "endstream", "/ObjStm", "/JS", "/JavaScript", "/AA", "/Launch", "/OpenAction", "/RichMedia"]
#                     0       1         2           3           4        5         6           7        10           8              9
# /ObjStm counts the number of object streams. An object stream is a stream object that can contain other objects, and can therefor be used to obfuscate objects (by using different filters).
# /JS and /JavaScript indicate that the PDF document contains JavaScript.
# Almost all malicious PDF documents that I've found in the wild contain JavaScript (to exploit a JavaScript vulnerability and/or to execute a heap spray).
# Of course, you can also find JavaScript in PDF documents without malicious intend.
# /AA and /OpenAction indicate an automatic action to be performed when the page/document is viewed.
# All malicious PDF documents with JavaScript I've seen in the wild had an automatic action to launch the JavaScript without user interaction.
# The combination of automatic action  and JavaScript makes a PDF document very suspicious.
# /RichMedia can imply presence of flash file.
# /Launch counts launch actions.
# /AcroForm this tag is defined if a document contains form fields, and is true if it uses XML Forms Architecture; not a real Tag ID


# global variables
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english')) 


class dataPDF:
    __filename = ""                                 # path to pdf file
    __image = ""                                    # path to image of pdf file
    __text = ""                                     # path to JavaScript from pdf file
    __shortname = ""                                # name of pdf file
    __histblur = []                                 # vector of color histogram and blur. Length = 513
    __text_tfidf = ""                               # correct text from first page or vector for tfidf
    __dsurlsjsentropy = []                          # vector of tags, urls, JavaScript and entropy. Length = 32
    __folder_path = ""                              # path to classes folder
    __isjs_path = "JaSt-master/js/"                 # path for is_js.py code
    __csv_path = ""
    # constructor
    def __init__(self, filename, folder_path, dataset):
        self.__folder_path = folder_path + "classes/"
        self.__isjs_path = folder_path + "classes/JaSt-master/js/"
        self.__filename = folder_path + dataset + "/" + filename
        self.__image = folder_path + "IMAGES/" + filename.replace('pdf', 'jpg')
        self.__text = folder_path + "TEXTS/" + filename + ".txt"
        self.__shortname = filename
        self.__csv_path = folder_path + "pdfFILES.csv"

    # this function extract color histogram for images
    def extract_color_histogram(self, image, bins=(8, 8, 8)):
        try:
            # extract a 3D color histogram from the HSV color space using
            # the supplied number of `bins` per channel
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,[0, 180, 0, 256, 0, 256])

            # handle normalizing the histogram if we are using OpenCV 2.4.X
            if imutils.is_cv2():
                hist = cv2.normalize(hist)

            # otherwise, perform "in place" normalization in OpenCV 3 (I
            # personally hate the way this is done
            else:
                cv2.normalize(hist, hist)

            # return the flattened histogram as the feature vector
            return hist.flatten()
        except Exception:
            hist = list([-1]*512)                         # -1, error. Using the * operator for initialization 
            return hist  


    # this function detect blur
    def detect_image_blur(self, imgPath):
        try:
            image = cv2.imread(imgPath)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            score = cv2.Laplacian(image, cv2.CV_64F).var()
            detect = {score}                                    # score < 110, an image is blur
            return detect
        except Exception:
            detect = {-1}                                       # -1, error
            return detect

    # calculate histogram and blur
    def calculate_histogram_blur(self):
        try:
            # load the image and extract the class label (assuming that our
            # path as the format: /path/to/dataset/{class}.{image_num}.jpg
            image = cv2.imread(self.__image)

            # histogram to characterize the color distribution of the pixels
            # in the image
            hist = self.extract_color_histogram(image)

            # detect blur
            blur = self.detect_image_blur(self.__image)
        
            hist = list(hist) + list(blur)
            self.__histblur = np.array(hist)
        except Exception:
            self.__histblur = list([-1]*513)                         # -1, error. Using the * operator for initialization 

    # this function clean text from garbage
    def clean_text(self, text):
        """
            text: a string  
            return: modified initial string
        """
        text = BeautifulSoup(text, "lxml").text                                   # HTML decoding
        text = text.lower()                                                       # lowercase text
        text = REPLACE_BY_SPACE_RE.sub(' ', text)                                 # replace REPLACE_BY_SPACE_RE symbols by space in text
        text = BAD_SYMBOLS_RE.sub('', text)                                       # delete symbols which are in BAD_SYMBOLS_RE from text
        text = ' '.join(word for word in text.split() if word not in STOPWORDS)   # delete stopwors from text
        return text
    
    # this function save clean text from pdf file
    def save_text(self, text):
        self.__text_tfidf = self.clean_text(text)
        with open(self.__csv_path, 'a') as csvFile:
            fields = ['File', 'Text']
            writer = csv.DictWriter(csvFile, fieldnames = fields)
            row = [{'File': self.__shortname, 'Text':self.__text_tfidf}]
            writer.writerows(row)
        csvFile.close()


    # function for part of JavaScript
    # Sources:
    # https://stackoverflow.com/questions/29342542/how-can-i-extract-a-javascript-from-a-pdf-file-with-a-command-line-tool
    # js extraction code
    # https://github.com/Aurore54F/JaSt
    # JAST project
    def pdfJS(self):
        # variables for features
        num_objects = 0
        num_js_lines = 0
        num_backslash = 0
        num_evals = 0
        type_js = 0  # no - 0, valid - 1, malformed - 2
        encoding = 0
    
        # handling the case that previous file failed to parse
        errorfile = os.path.isfile(self.__text)                                            # holds boolean value
        if not errorfile:
            print(self.__shortname  + " failed parsing!")
            features = [-1, -1, -1, -1, -1]
            return features
        else:
            temp_file = open(self.__text, 'r')
            # copy content from temp file to text file
            try:
                for line in str(temp_file.readlines()):
                    if "// peepdf comment: Javascript code located in object" in line:
                        num_objects = num_objects + 1
                    elif line != '\n':
                        num_js_lines = num_js_lines + 1
                    # string literal for backslash
                    num_backslash = num_backslash + line.count("\\")
                    num_evals = num_evals + line.count("eval")
            except:
                encoding = -1
            temp_file.close()

            # check if valid JS or malformed JS
            if num_js_lines != 0:
                isjs = subprocess.Popen(['python', self.__isjs_path + "is_js.py", "--f", self.__text], stdout=subprocess.PIPE)
                isjs.wait()
                for line in isjs.stdout:
                    if "malformed" in str(line):
                        type_js = 2
                    elif " valid" in str(line):
                        type_js = 1

            # save and print features
            features = [num_objects, num_js_lines, num_backslash, num_evals, type_js]
            return features

    # function for part of Entropy
    # ans[0] - total_entropy; ans[1] - entropy_inside; ans[2] - entropy_outside
    # Source: https://github.com/hiddenillusion/AnalyzePDF
    def entropy(self):
        try:
            ans = []
            p = subprocess.Popen(['python', self.__folder_path + 'AnalyzePDF-master/AnalyzePDF.py', self.__filename], stdout=subprocess.PIPE)
            p.wait()
            for (i, line) in enumerate(p.stdout):
                line = str(line)
                pattern = r"(\d+.\d+)"
                num = re.search(pattern, line).group()
                ans.append(float(num))
                if(i == 2):
                    break
            return ans
        except Exception:
            ex = [-1, -1, -1]
            return ex

    # function for part of pdfid.py
    def defaultJS(self):
        try:
            ans = []
            p = subprocess.Popen(['python', self.__folder_path + 'pdfid_v0_2_5/pdfid.py', self.__filename], stdout=subprocess.PIPE)
            p.wait()
            for line in p.stdout:
                line = str(line)
                if '%PDF' in line or line.startswith('PDFiD'):
                    continue
                pattern1 = r"\s*(\S+)\s+(\d+)"
                m = re.search(pattern1, line)
                if m is not None:
                    key = m.group(1)
                    if key in default_features:
                        ans.append(int(m.group(2)))
            return ans
        except Exception:
            ex = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
            return ex

    # function for part of URLs
    def URLs(self):
        try:
            p = subprocess.Popen(['python', self.__folder_path + 'support_union.py', self.__filename], stdout = subprocess.PIPE)
            p.wait()
            out, err = p.communicate()
            out = str(out)
            out = out.replace('b\'','').replace('\\n\'','').replace('[','').replace(']','').split(',')
            if ('-1' in out[0]):
                return list(map(int, out))
            out = list(map(int, out))
            return out
        except Exception:
            ex = [-1, -1, -1, -1, -1, -1, -1, -1, -1]
            return ex

    # this function calculate vector __dsurlsjsentropy
    def calculate_dsurlsjsentropy(self):
        # tags
        ans = self.defaultJS()
        # urls
        urls = self.URLs()
        # JavaScript
        js = self.pdfJS()
        # entropy
        entropies = self.entropy()
        # union
        ans = ans + urls
        ans = ans + js
        ans = ans + entropies
        self.__dsurlsjsentropy = np.array(ans)

    # this function return filename
    def getFilename(self):
        return self.__filename

    # this function return short filename
    def getShortname(self):
        return self.__shortname

    # this function return path to image
    def getImage(self):
        return self.__image

    # this function returns object histogram
    def getImgHistogram(self):
        return self.__histblur

    # this function returns object feature vector
    def getFeatVec(self):
        return self.__dsurlsjsentropy

    # this function returns object text
    def getText(self):
        return self.__text_tfidf

    # this function remove image
    def removeIMAGE(self):
        os.remove(self.__image)

    # print all information
    def printData(self):
        print(self.__filename)
        print(self.__image)
        print(self.__text)
        print(self.__shortname)
        print(self.__histblur)
        print(self.__text_tfidf)
        print(self.__dsurlsjsentropy)
