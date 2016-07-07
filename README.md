1. This is a one day work to replicate a document quality estimation project, the paper is
   "http://lampsrv02.umiacs.umd.edu/pubs/Papers/lekang14-DocIQA/lekang14-DocIQA.pdf"
2. First, document image of very large size (3264 X 1840) is preprocessed in the following ways:
   - Local normalisation using a 5 X 5 filter
   - Otsu binarization and further binarization to extract text/non-text regions
   - Extraction of 850, 48 X 48 patches from each document image.
3. Each patch is assigned a score that corresponds to the document image itself (weird, I know!!)
4. Further CNN is trained (model is made with keras detailed in regression.py).
5. Linear Correlation Coefficient and Spearmans Rank Order Correlation Coefficient is found in between
   predicted and given scores.
6. State-of-the-art results, LCC:0.95 and SROCC:0.923.
7. The leaned weights are stored in weights.hdf5
8. splits.hdf5 contained training and testing data. The validation data is created from training data in
   the data.py file. document.hdf5 file contains the complete dataset including the labels. From docmuent.hdf5
   dataset training, testing and validation data is extracted.
9. Other details related to the architecture can be found in the regression.py file itself.

NOTE: To run the code using GPU, write the following command in the terminal
      THEANO_FLAGS=device=gpu,floatX=float32 python my_keras_script.py

## Experiment: Training on 20 document groups and testing on 5 document groups containing 35 images.
   - Results per image level:
     LCC = 0.7871
     SROCC = 0.75404
   - For all the patches as a whole:
     LCC = 0.4636
     SROCC = 0.45
   With 'mae' loss, the results are:
   - Results per image level:
     LCC = 0.782
     SROCC = 0.7527
   - For all the patches as a whole:
     LCC = 0.498
     SROCC = 0.478
   With random shuffling using scikit crossvalidation function, the results are:
     LCC = 0.88872
     SROCC = 0.872294
   - For all the patches as a whole:
     LCC = 0.632
     SROCC = 0.611

   With random shuffling using scikit crossvalidation function and using comeplte training dataset, the results are:
   - Test Results per image level:
     LCC = 0.89872
     SROCC = 0.877294
   - Test results for all the pacthes as a whole::
     LCC = 0.644
     SROCC = 0.6211

   With test data provided by Prof. Gandhi, results are following:
   - Test Results per image level:
     LCC: 0.42
     SROCC: 0.71335
   - Test results for all the pacthes as a whole:
     LCC = 0.4135
     SROCC = 0.6705

   The data for the test is stored in /data4/gopal.sharma/image_quality/test_data
   The code to test the real data (unknown) is in ~/codes/Image_Quality/real_test.py




