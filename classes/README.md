In order to prepare all the data for our project, we have united the extraction code to three main files (in classes folder) and imported these classes to the main file (as.py - in Phase5 folder):
 * createDATA.py - Extract image of first page, and extract JS from objects.
 * dataPDF.py - Creates the vectors for image and feature machines (image vector, feature vector).\
 The text vector will be created in the main file.
 * readPDF.py - Extracts text from the first page, and in case of error, extracts the text from the image of the first page.
 <br>
 
PLEASE NOTE: support_union.py - Helps dataPDF.py create the feature vector.

<br>

In order to run the files, you will need to download these folders:

  * download: `AnalyzePDF-master`
  * download: `JaSt-master`
  * download: `node_modules`
  * download: `pdfid_v*`
  * download: `peepdf`
