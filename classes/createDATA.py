# -------------------------------------------------------------------------------------------------------------------------------------------------------------
# The our class for PDF files
# @Authors:  Alexey Titov and Shir Bentabou
# @Version: 1.0
# -------------------------------------------------------------------------------------------------------------------------------------------------------------

# libraries
import cv2
import os
from pdf2image import convert_from_path
from PyPDF2 import PdfFileReader
from imutils import paths
import numpy as np
import pytesseract
import tempfile
import subprocess
import sys
import PyPDF2 as pyPdf
import random

class createDATA:
   
    __train = ""                              # folder of train files
    __folder_path = ""                        # path to main folder
    __images_path = ""                        # path to folder of images from first page
    # path to folder of JavaScript from pdf file
    __texts_path = ""
    # dictionary for translate PDF language to tessaract language
    __lan_lst = {
        "en-us": "eng",
        "en": "eng",
        "en-za": "eng",
        "en-gb": "eng",
        "en-in": "eng",
        "es-co": "spa",
        "es": "spa",
        "de-de": "deu",
        "fr-fr": "fra",
        "fr-ca": "fra"}

    # dictionary for /Root/Lang 1 - except; 2 - a file have not /Root/Lang; 3
    # - /Root/Lang = ''; 4 - language
    __ans_list = dict()

    # constructor
    def __init__(self, folder_path, train):
        self.__train = train
        self.__folder_path = folder_path
        self.__images_path = folder_path + "/IMAGES"
        self.__texts_path = folder_path + "/TEXTS"

    # this function update ans_list
    def add_ans_list(self, save_dir, base_filename, filename):
        try:
            name = os.path.join(save_dir, base_filename)
            pdfFile = PdfFileReader(open(filename, 'rb'), strict = False)
            catalog = pdfFile.trailer['/Root'].getObject()
            if "/Lang" in catalog:
                lang = catalog['/Lang'].getObject()
                if (lang == ''):
                    self.__ans_list.update({name: [3, 'None']})
                else:
                    lang = lang.lower()
                    language = self.__lan_lst.get(lang)
                    self.__ans_list.update({name: [4, language]})
            else:
                self.__ans_list.update({name: [2, 'None']})
        except Exception as ex:
            #print(filename)
            #print(ex)
            self.__ans_list.update({name: [1, 'None']})

    # this function generate random page
    def get_page_random(self, pdf):
        try:
            page_start = pdf.trailer["/Root"]["/PageLabels"]["/Nums"][1].getObject()["/St"]
        except:
            page_start = 1
        page_count = pdf.getNumPages()
        page_stop = page_start + page_count
        random_page = random.randint(page_start, page_stop)
        return random_page

    # this function convert pdf file to jpg file
    def convert(self, dirpdf):
        # dir of folder and filter for pdf files
        files = [f for f in os.listdir(dirpdf) if os.path.isfile(os.path.join(dirpdf, f))]
        files = list(filter(lambda f: f.endswith(('.pdf', '.PDF')), files))

        # variables for print information
        cnt_files = len(files)
        i = 0
        for filepdf in files:
            try:
                filename = os.path.join(dirpdf, filepdf)
                images_from_path = 0
                with tempfile.TemporaryDirectory() as path:
                    pdf = pyPdf.PdfFileReader(open(filename, "rb"))
                    random_page = self.get_page_random(pdf)
                    images_from_path = convert_from_path(filename, output_folder=path, last_page=random_page, first_page=random_page-1)

                base_filename = os.path.splitext(os.path.basename(filename))[0] + '.jpg'

                # save image
                for page in images_from_path:
                    page.save(os.path.join(self.__images_path, base_filename), 'JPEG')
                i += 1

                # update ans_list
                self.add_ans_list(self.__images_path, base_filename, filename)

                # show an update every 50 images
                if (i > 0 and i % 50 == 0):
                    print("[INFO] processed {}/{}".format(i, cnt_files))
            except Exception:
                # always keep track the error until the code has been clean
                print("[!] Convert PDF to JPEG " + filepdf)
                continue
                return False
        print("[INFO] processed {}/{}".format(cnt_files, cnt_files))
        return True

    # this function convert pdf file to image
    def convertOnePDF(self, filepdf):
        try:
            filename = os.path.join(self.__folder_path, filepdf)
            images_from_path = 0
            with tempfile.TemporaryDirectory() as path:
                pdf = pyPdf.PdfFileReader(open(filename, "rb"))
                random_page = self.get_page_random(pdf)
                images_from_path = convert_from_path(filename, output_folder=path, last_page=random_page, first_page=random_page-1)

            base_filename = os.path.splitext(os.path.basename(filename))[0] + '.jpg'

            # save image
            for page in images_from_path:
                page.save(os.path.join(self.__images_path, base_filename), 'JPEG')
                
            # update ans_list
            self.add_ans_list(self.__images_path, base_filename, filename)
            return True
        except Exception as ex:
            # print(ex)
            # always keep track the error until the code has been clean
            print("[!] Convert PDF to JPEG " + filepdf)
            return False

    # this function extract JavaScript from pdf file to txt
    def extractOnePDF(self, filepdf):
        err_path = os.path.join(self.__folder_path, 'classes/peepdf/errors.txt')
        try:
            filename = os.path.join(self.__folder_path, filepdf)
            self.ex_js(filename)
            # handling the case that previous file failed to parse
            errorfile = os.path.isfile(err_path)                                          # holds boolean value
            if errorfile:
                os.remove(err_path)
            else:
                fi_na = open(self.__texts_path + '/' + str(filepdf) +'.txt', 'w+')        # open text file for current file
                temp_file = open(self.__folder_path + '/classes/JSfromPDF.txt', 'r')
                # copy content from temp file to text file
                for line in temp_file.readlines():
                   try:     
                       fi_na.write(str(line))
                   except:
                       continue
                temp_file.close()
                fi_na.close()
            return True
        except Exception as ex:
            # print(ex)
            # always keep track the error until the code has been clean
            print("[!] Extract JavaScript to TXT " + filepdf)
            return False


    # function extract JS from pdf using peepdf
    def ex_js(self, filename):
        # run peepdf.py
        scriptpath = os.path.join(self.__folder_path + '/classes/peepdf/', 'peepdf.py')               # joins to path
        CommandOfExtract = self.__folder_path + '/classes/extractJS.txt'
        p = subprocess.Popen(['python', scriptpath, '-l', '-f', '-s', CommandOfExtract, filename])    # open subprocess and extract
        p.wait()


    # this function extract JavaScript from pdf file to txt
    def extract(self, dirpdf):
        # dir of folder and filter for pdf files
        files = [f for f in os.listdir(dirpdf) if os.path.isfile(os.path.join(dirpdf, f))]
        files = list(filter(lambda f: f.endswith(('.pdf', '.PDF')), files))

        # variables for print information
        cnt_files = len(files)
        i = 0
        err_path = os.path.join(self.__folder_path, 'classes/peepdf/errors.txt')
        for filepdf in files:
            i += 1
            try:
                filename = os.path.join(dirpdf, filepdf)
                self.ex_js(filename)
                # handling the case that previous file failed to parse
                errorfile = os.path.isfile(err_path)                                          # holds boolean value
                if errorfile:
                    os.remove(err_path)
                else:
                    fi_na = open(self.__texts_path + '/' + str(filepdf) +'.txt', 'w+')        # open text file for current file
                    temp_file = open(self.__folder_path + '/classes/JSfromPDF.txt', 'r')
                    # copy content from temp file to text file
                    try:
                        for line in temp_file.readlines():
                            fi_na.write(str(line))
                    except:
                        continue
                    temp_file.close()
                    fi_na.close()
                # show an update every 50 pdf files
                if (i > 0 and i % 50 == 0):
                    print("[INFO] processed {}/{}".format(i, cnt_files))
            except Exception as ex:
                #print(ex)
                # always keep track the error until the code has been clean
                print("[!] Extract JavaScript to TXT " + filepdf)
                return False
        print("[INFO] processed {}/{}".format(cnt_files, cnt_files))
        return True

    # this function return dictionary
    def getDict(self):
        return self.__ans_list
