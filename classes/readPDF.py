# -------------------------------------------------------------------------------------------------------------------------------------------------------------
# The our class for PDF files
# @Authors:  Alexey Titov and Shir Bentabou
# @Version: 1.0
# -------------------------------------------------------------------------------------------------------------------------------------------------------------

# libraries
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import XMLConverter, HTMLConverter, TextConverter
from pdfminer.layout import LAParams
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns
from pdf2image import convert_from_path
from PyPDF2 import PdfFileReader
import imutils
import cv2
import pytesseract
import sys
import os
import tempfile
import PyPDF2 as pyPdf
import random

# fix UnicodeEncodeError
if sys.version[0] == '2':
    reload(sys)
    sys.setdefaultencoding("utf-8")


class readPDF:
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
    def __init__(self, ans_list):
        self.__ans_list = ans_list

    # this function read information from image
    def extract_text_image(self, imgPath):
        # Define config parameter
        # '--oem 1' for using LSTM OCR Engine
        config = ('--oem 1 --psm 3')

        # Read image from disk
        img = cv2.imread(imgPath, cv2.IMREAD_COLOR)

        # Read /Root/Lang
        values = self.__ans_list.get(imgPath)
        try:
            if (values[0] == 4):
                langs = value[1]
                imagetext = pytesseract.image_to_string(img, lang=langs, config=config)
            else:
                imagetext = pytesseract.image_to_string(img, config=config)
            return imagetext
        except Exception as ex:
            print(imgPath)
            print(ex)
            imagetext = "except"
            return imagetext

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

    # this function extract text from pdf
    def extractTEXT(self, filename, imagename):
        fp = open(filename, 'rb')
        try:
            pdf = pyPdf.PdfFileReader(fp)
            random_page = self.get_page_random(pdf)
        except:
            random_page = 1
        try:
            os.chmod(imagename,0o777)
            os.remove(imagename)
        except OSError:
            pass
        pagenos = set()
        data = ""
        rsrcmgr = PDFResourceManager()
        retstr = StringIO()
        laparams = LAParams()
        device = TextConverter(rsrcmgr, retstr, codec='utf-8', laparams=laparams)
        # create a PDF interpreter object
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        # Process each page contained in the document
        try:
            index = 0
            for page in PDFPage.get_pages(fp, pagenos, maxpages=0, password="", caching=True, check_extractable=False):
                index += 1
                if (index != random_page):
                    continue
                interpreter.process_page(page)
                data = retstr.getvalue()
                # number of character is anomaly
                if (len(data) < 2 or len(data) > 60000):
                    try:
                        images_from_path = 0
                        with tempfile.TemporaryDirectory() as path:
                            images_from_path = convert_from_path(filename, output_folder=path, last_page=random_page, first_page=random_page-1)

                        # save image
                        for page in images_from_path:
                            page.save(imagename, 'JPEG')
                        data = self.extract_text_image(imagename)
                    except Exception as ex:
                        print("[!] Convert PDF to JPEG " + filename)
                        data = 'except'
                break
        except Exception as ex:
            print(ex)
            print(filename)
            try:
                images_from_path = 0
                with tempfile.TemporaryDirectory() as path:
                    images_from_path = convert_from_path(filename, output_folder=path, last_page=random_page, first_page=random_page-1)

                # save image
                for page in images_from_path:
                    page.save(imagename, 'JPEG')
                data = self.extract_text_image(imagename)
            except Exception as ex:
                print("[!] Convert PDF to JPEG " + filename)
                data = 'except'
        # Cleanup
        device.close()
        retstr.close()
        return data
