import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage.feature import hog

import os

path = os.listdir('brain_tumor/Training/')
classes = {'no_tumor':0, 'pituitary_tumor':1}


import cv2
X = []
Y = []
for cls in classes:
    pth = 'brain_tumor/Training/'+cls
    for j in os.listdir(pth):
        img = cv2.imread(pth+'/'+j, 0)
        img = cv2.resize(img, (200,200))
        X.append(img)
        Y.append(classes[cls])
        
        
X = np.array(X)
print(X)
Y = np.array(Y)
X_updated = X.reshape(len(X), -1)

np.unique(Y)

pd.Series(Y).value_counts()

X.shape, X_updated.shape


plt.imshow(X[0], cmap='gray')

X_updated = X.reshape(len(X), -1)
X_updated.shape

xtrain, xtest, ytrain, ytest = train_test_split(X_updated, Y, random_state=10,
                                               test_size=.20)

xtrain.shape, xtest.shape

print(xtrain.max(), xtrain.min())
print(xtest.max(), xtest.min())
xtrain = xtrain/255
xtest = xtest/255
print(xtrain.max(), xtrain.min())
print(xtest.max(), xtest.min())

from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import warnings
warnings.filterwarnings('ignore')

lg = LogisticRegression(C=0.1)
lg.fit(xtrain, ytrain)


sv = SVC()
sv.fit(xtrain, ytrain)

print("Training Score:", lg.score(xtrain, ytrain))
print("Testing Score:", lg.score(xtest, ytest))

print("Training Score:", sv.score(xtrain, ytrain))
print("Testing Score:", sv.score(xtest,ytest))

pred = sv.predict(xtest)

misclassified=np.where(ytest!=pred)
misclassified

print("Total Misclassified Samples: ",len(misclassified[0]))
print(pred[36],ytest[36])

dec = {0:'No Tumor', 1:'Positive Tumor'}

plt.figure(figsize=(12,8))
p = os.listdir('brain_tumor/Testing/')
c=1
for i in os.listdir('brain_tumor/Testing/no_tumor/')[:9]:
    plt.subplot(3,3,c)
    
    img = cv2.imread('brain_tumor/Testing/no_tumor/'+i,0)
    img1 = cv2.resize(img, (200,200))
    img1 = img1.reshape(1,-1)/255
    #_,hog=hog(img,orientations=8,pixels_per_cell=(16,16),cells_per_block=(1,1),visualize=True,multichannel=True)
    p = sv.predict(img1)
    plt.title(dec[p[0]])
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    c+=1
    
plt.figure(figsize=(12,8))
p = os.listdir('brain_tumor/Testing/')
c=1
for i in os.listdir('brain_tumor/Testing/pituitary_tumor/')[:16]:
    plt.subplot(4,4,c)
   
    img = cv2.imread('brain_tumor/Testing/pituitary_tumor/'+i,0)
    img1 = cv2.resize(img, (200,200))
    img1 = img1.reshape(1,-1)/255
    p = sv.predict(img1)
    plt.title(dec[p[0]])
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    c+=1
    
    
    
############################################    
    
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QIcon
import numpy as np
global a
import glob, os, os.path


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(900,600)
        MainWindow.setStyleSheet("background-color: #1B2838;")
        MainWindow.setWindowIcon(QIcon("icon.png"))
        
        #center frame
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(20, 55, 781, 641))
        self.frame.setStyleSheet("background-color: qlineargradient(spread:pad, x1:0.630318, y1:0.392, x2:0.068, y2:0, stop:0.551136 #1B698F, stop:1 #1B698F);\n"
                                 "   border-radius: 25px;\n"
                                 "    border: 2px solid #EA3B15;")
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        
        self.frame1 = QtWidgets.QFrame(self.centralwidget)
        self.frame1.setGeometry(QtCore.QRect(900, 300, 400, 200))
        self.frame1.setStyleSheet("background-color:qlineargradient(spread:pad, x1:0.630318, y1:0.392,  x2:0.068, y2:0, stop:0.551136 #ffffff, stop:1 #ffffff);\n"
                                 "   border-radius: 25px;\n"
                                 "    border: 2px solid #EA3B15;")
        self.frame1.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame1.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame1.setObjectName("frame1")
#########        
        self.slctimg = QtWidgets.QPushButton(self.frame)
        self.slctimg.setGeometry(QtCore.QRect(600, 10, 171, 61))
        self.slctimg.setStyleSheet("color:white;\n"
                                   "font: 14pt \"Gadugi\";\n"
                                   "   border-radius: 20px;\n"
                                   "    border: 2px solid #00c6fb;\n"
                                   "background-color:#1B2838;\n"
                                   "width:171;\n"
                                   "height:61")
        self.slctimg.setObjectName("slctimg")
        
        self.rgbtgray = QtWidgets.QPushButton(self.frame)
        self.rgbtgray.setGeometry(QtCore.QRect(600, 80, 171, 61))
        self.rgbtgray.setStyleSheet("color:white;\n"
                                    "font: 14pt \"Gadugi\";\n"
                                    "   border-radius: 20px;\n"
                                    "    border: 2px solid #00c6fb;\n"
                                    "background-color:#1B2838;\n"
                                    "width:171px;\n"
                                    "height:61px;")
        self.rgbtgray.setObjectName("rgbtgray")
        
        self.bilfil = QtWidgets.QPushButton(self.frame)
        self.bilfil.setGeometry(QtCore.QRect(600, 150, 171, 61))
        self.bilfil.setStyleSheet("color:white;\n"
                                  "font: 14pt \"Gadugi\";\n"
                                  "   border-radius: 20px;\n"
                                  "    border: 2px solid #00c6fb;\n"
                                  "background-color:#1B2838;\n"
                                  "width:171px;\n"
                                  "height:61px;")
        self.bilfil.setObjectName("bilfil")
        
        self.medfil = QtWidgets.QPushButton(self.frame)
        self.medfil.setGeometry(QtCore.QRect(600, 220, 171, 61))
        self.medfil.setStyleSheet("color:white;\n"
                                  "font: 14pt \"Gadugi\";\n"
                                  "   border-radius: 20px;\n"
                                  "    border: 2px solid #00c6fb;\n"
                                  "background-color:#1B2838;\n"
                                  "width:171px;\n"
                                  "height:61px;")
        self.medfil.setObjectName("medfil")
        
        self.gaufil = QtWidgets.QPushButton(self.frame)
        self.gaufil.setGeometry(QtCore.QRect(390, 150, 111, 41))
        self.gaufil.setStyleSheet("background-color: #008CBA;\n"
                                  "text-align: center;\n"
                                  "\n"
                                  "")
        self.gaufil.setObjectName("gaufil")
        
        self.thres = QtWidgets.QPushButton(self.frame)
        self.thres.setGeometry(QtCore.QRect(600, 290, 171, 61))
        self.thres.setStyleSheet("color:white;\n"
                                 "font: 14pt \"Gadugi\";\n"
                                 "   border-radius: 20px;\n"
                                 "    border: 2px solid #00c6fb;\n"
                                 "background-color:#1B2838;\n"
                                 "width:171px;\n"
                                 "height:61px;")
        self.thres.setObjectName("thres")
        
        self.dilate = QtWidgets.QPushButton(self.frame)
        self.dilate.setGeometry(QtCore.QRect(600, 360, 171, 61))
        self.dilate.setStyleSheet("color:white;\n"
                                  "font: 14pt \"Gadugi\";\n"
                                  "   border-radius: 20px;\n"
                                  "    border: 2px solid #00c6fb;\n"
                                  "background-color:#1B2838;\n"
                                  "width:171px;\n"
                                  "height:61px;")
        self.dilate.setObjectName("dilate")
        
        self.morpho = QtWidgets.QPushButton(self.frame)
        self.morpho.setGeometry(QtCore.QRect(600, 430, 171, 61))
        self.morpho.setStyleSheet("color:white;\n"
                                  "font: 14pt \"Gadugi\";\n"
                                  "   border-radius: 20px;\n"
                                  "    border: 2px solid #00c6fb;\n"
                                  "background-color:#1B2838;\n"
                                  "width:171px;\n"
                                  "height:61px;")
        self.morpho.setObjectName("morpho")
        
        self.addcol = QtWidgets.QPushButton(self.frame)
        self.addcol.setGeometry(QtCore.QRect(600, 500, 171, 61))
        self.addcol.setStyleSheet("color:white;\n"
                                  "font: 14pt \"Gadugi\";\n"
                                  "   border-radius: 20px;\n"
                                  "    border: 2px solid #00c6fb;\n"
                                  "background-color:#1B2838;\n"
                                  "width:171px;\n"
                                  "height:61px;")
        self.addcol.setObjectName("addcol")
        
        self.savimg = QtWidgets.QPushButton(self.frame)
        self.savimg.setGeometry(QtCore.QRect(600, 570, 171, 61))
        self.savimg.setStyleSheet("color:white;\n"
                                  "font: 14pt \"Gadugi\";\n"
                                  "   border-radius: 20px;\n"
                                  "    border: 2px solid #00c6fb;\n"
                                  "background-color:#1B2838;\n"
                                  "width:171px;\n"
                                  "height:61px;")
        self.savimg.setObjectName("savimg")
        
        self.output = QtWidgets.QPushButton(self.frame1)
        self.output.setGeometry(QtCore.QRect(0, 0, 400, 200))
        self.output.setStyleSheet("color:white;\n"
                                  "font: 14pt \"Gadugi\";\n"
                                  "   border-radius: 20px;\n"
                                  "    border: 2px solid #00c6fb;\n"
                                  "background-color:#FFFFFF;\n"
                                  "width:171px;\n"
                                  "height:61px;")
        self.output.setObjectName("output")
        
        
        
 #########       
        self.label = QtWidgets.QLabel(self.frame)
        self.label.setGeometry(QtCore.QRect(20, 90, 261, 421))
        self.label.setStyleSheet("background-color: rgb(218, 218, 218);\n"
                                 "   border-radius: 20px;\n"
                                 "    border: 2px solid #00c6fb;")
        self.label.setText("")
        self.label.setObjectName("label")
        
        self.label_2 = QtWidgets.QLabel(self.frame)
        self.label_2.setGeometry(QtCore.QRect(299, 90, 261, 421))
        self.label_2.setStyleSheet("background-color: rgb(218, 218, 218);\n"
                                   "   border-radius: 20px;\n"
                                   "    border: 2px solid #00c6fb;")
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")
         
        
        self.titlelbl = QtWidgets.QLabel(self.centralwidget)
        self.titlelbl.setGeometry(QtCore.QRect(450, 10, 501, 41))
        font = QtGui.QFont()
        font.setFamily("MS Shell Dlg 2")
        font.setPointSize(26)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.titlelbl.setFont(font)
        self.titlelbl.setStyleSheet("opacity:0.6;\n"
                                    "font: 26pt \"MS Shell Dlg 2\";\n"
                                    "color: rgb(216, 216, 216);")
        self.titlelbl.setObjectName("titlelbl")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)


        error_dialog = QtWidgets.QErrorMessage()
        error_dialog.setWindowTitle('Image processing tool')
        error_dialog.setWindowIcon(QIcon('"C:/Users/DELL/Desktop/BrainTumorSegmentation-PYTHON-master/"'))
        error_dialog.showMessage('Please click the buttons in Sequential order to proceed!')
        error_dialog.exec()

        path = "C:/Users/DELL/Desktop/BrainTumorSegmentation-PYTHON-master/"

        filelist = glob.glob(os.path.join(path, "*.png"))
        for f in filelist:
            os.remove(f)
            print('deleted')
#button call
        self.slctimg.clicked.connect(self.setImage)
        self.rgbtgray.clicked.connect(self.filter1)
        self.bilfil.clicked.connect(self.filter2)
        self.medfil.clicked.connect(self.filter3)
        self.thres.clicked.connect(self.threshing)
        self.dilate.clicked.connect(self.dilation)
        self.morpho.clicked.connect(self.contouring)
        self.addcol.clicked.connect(self.applyingcolor)
        self.savimg.clicked.connect(self.saving)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Tumor Segmentation Panel"))
        self.slctimg.setText(_translate("MainWindow", "Select Image"))
        self.rgbtgray.setText(_translate("MainWindow", "Bilateral Filter"))
        self.bilfil.setText(_translate("MainWindow", "Median Filter"))
        self.medfil.setText(_translate("MainWindow", "Gaussian Filter"))
        #self.gaufil.setText(_translate("MainWindow", "Gaussian Filter"))
        self.thres.setText(_translate("MainWindow", " Thresholding"))
        self.dilate.setText(_translate("MainWindow", "Dilation"))
        self.morpho.setText(_translate("MainWindow", "Morphology"))
        self.addcol.setText(_translate("MainWindow", "Add color map"))
        self.savimg.setText(_translate("MainWindow", "Save Image"))
        self.titlelbl.setText(_translate("MainWindow", "Tumor Segmentation Panel"))

    def setImage(self):
        
        import cv2
        global a
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select Image", "C:/Users/DELL/Desktop/brain-tumor-detection-master/brain_tumor/Testing",
                                                            "Image Files (*.png *.jpg *jpeg *.bmp)")  # Ask for file

        a = fileName

        if fileName:  # If the user gives a file

            pixmap = QtGui.QPixmap(fileName)  # Setup pixmap with the provided image
            pixmap = pixmap.scaled(self.label.width(), self.label.height(),
                                   QtCore.Qt.KeepAspectRatio)  # Scale pixmap
            self.label.setPixmap(pixmap)  # Set the pixmap onto the label
            self.label.setAlignment(QtCore.Qt.AlignCenter)  # Align the label to center
            
            


            img = cv2.imread(a,0)
            img1 = cv2.resize(img, (200,200))
            img1 = img1.reshape(1,-1)/255
            p = sv.predict(img1)
            #self.label_3.setText(dec[p[0]])
            #self.frame1.setText(dec[p[0]])
            
            
            _translate = QtCore.QCoreApplication.translate
            self.output.setText(_translate("MainWindow",dec[p[0]] ))
            if(p[0]==0):
                self.output.setStyleSheet("color:white;\n"
                                      "font: 14pt \"Gadugi\";\n"
                                      "   border-radius: 20px;\n"
                                      "    border: 2px solid #00c6fb;\n"
                                      "background-color:#00FF00;\n"
                                      "width:171px;\n"
                                      "height:61px;")
                
            if(p[0]==1):
                self.output.setStyleSheet("color:white;\n"
                                      "font: 14pt \"Gadugi\";\n"
                                      "   border-radius: 20px;\n"
                                      "    border: 2px solid #00c6fb;\n"
                                      "background-color:#FF0000;\n"
                                      "width:171px;\n"
                                      "height:61px;")

    def filter1(self):
        import cv2
        import os


        img = cv2.imread(a)

        grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        path = "C:/Users/DELL/Desktop/brain-tumor-detection-master/finaldata/"
        cv2.imwrite(os.path.join(path, 'newtest.png'), grey)

        img = cv2.imread(a)
        bilateral = cv2.bilateralFilter(img, 2, 50, 50)

        path = "C:/Users/DELL/Desktop/brain-tumor-detection-master/finaldata/"
        cv2.imwrite(os.path.join(path, 'bilateral.png'), bilateral)

        pixmap = QtGui.QPixmap(os.path.join(path,'bilateral.png'))  # Setup pixmap with the provided image
        pixmap = pixmap.scaled(self.label_2.width(), self.label_2.height(),
                               QtCore.Qt.KeepAspectRatio)  # Scale pixmap
        self.label_2.setPixmap(pixmap)  # Set the pixmap onto the label

    def filter2(self):
        import cv2
        import os

        path = "C:/Users/DELL/Desktop/brain-tumor-detection-master/finaldata/"
        img = cv2.imread(os.path.join(path, 'bilateral.png'))

        median = cv2.medianBlur(img, 5)

        cv2.imwrite(os.path.join(path, 'median.png'), median)

        pixmap = QtGui.QPixmap(os.path.join(path, 'median.png'))  # Setup pixmap with the provided image

        pixmap = pixmap.scaled(self.label_2.width(), self.label_2.height(),
                               QtCore.Qt.KeepAspectRatio)  # Scale pixmap
        self.label_2.setPixmap(pixmap)  # Set the pixmap onto the label

    def filter3(self):
        import cv2
        import os

        path = "C:/Users/DELL/Desktop/brain-tumor-detection-master/finaldata/"
        img = cv2.imread(os.path.join(path, 'median.png'))

        equ1 = cv2.GaussianBlur(img, (5, 5), cv2.BORDER_CONSTANT)

        cv2.imwrite(os.path.join(path, 'gaussian.png'), equ1)

        pixmap = QtGui.QPixmap(os.path.join(path, 'gaussian.png'))  # Setup pixmap with the provided image
        pixmap = pixmap.scaled(self.label_2.width(), self.label_2.height(),
                               QtCore.Qt.KeepAspectRatio)  # Scale pixmap
        self.label_2.setPixmap(pixmap)  # Set the pixmap onto the label

    def threshing(self):
        import cv2
        import os

        path = "C:/Users/DELL/Desktop/brain-tumor-detection-master/finaldata/"
        img = cv2.imread(os.path.join(path, 'gaussian.png'))

        ret, thresh1 = cv2.threshold(img, 127, 190, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)

        cv2.imwrite(os.path.join(path, 'thresh.png'), thresh1)

        pixmap = QtGui.QPixmap(os.path.join(path, 'thresh.png'))  # Setup pixmap with the provided image
        pixmap = pixmap.scaled(self.label_2.width(), self.label_2.height(),
                               QtCore.Qt.KeepAspectRatio)  # Scale pixmap
        self.label_2.setPixmap(pixmap)  # Set the pixmap onto the label

    
    def dilation(self):
        import cv2
        import os
        

        path = "C:/Users/DELL/Desktop/brain-tumor-detection-master/finaldata/"
        img = cv2.imread(a)

        kernel = np.ones((5, 5), np.uint8)
        opening1 = cv2.dilate(img, kernel, iterations=0)
        opening2 = cv2.morphologyEx(opening1, cv2.MORPH_CLOSE, kernel)

        TE=cv2.imwrite(os.path.join(path, 'dilated.png'), opening2)

        pixmap = QtGui.QPixmap(os.path.join(path, 'dilated.png'))  # Setup pixmap with the provided image

        pixmap = pixmap.scaled(self.label_2.width(), self.label_2.height(),
                               QtCore.Qt.KeepAspectRatio)  # Scale pixmap
        self.label_2.setPixmap(pixmap)  # Set the pixmap onto the label
    
    def contouring(self):
        import cv2
        import os
          
        path = "C:/Users/DELL/Desktop/brain-tumor-detection-master/finaldata/"
        img = cv2.imread(os.path.join(path, 'dilated.png'))
        grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        
        dilate = cv2.imread(os.path.join(path, "dilated.png"))
        grey2 = cv2.cvtColor(dilate, cv2.COLOR_RGB2GRAY)


        contours, hierarchy = cv2.findContours(grey2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        b = cv2.drawContours(grey, contours, -1, (0, 0, 255), 3)

        dst = cv2.addWeighted(grey2, 0.7, b, 0.3, 0)

        cv2.imwrite(os.path.join(path, 'contour.png'), dst)

        pixmap = QtGui.QPixmap(os.path.join(path, 'contour.png'))  # Setup pixmap with the provided image
        pixmap = pixmap.scaled(self.label_2.width(), self.label_2.height(),
                               QtCore.Qt.KeepAspectRatio)  # Scale pixmap
        self.label_2.setPixmap(pixmap)  # Set the pixmap onto the label

    def applyingcolor(self):
        import cv2
        import os
        
        
        path = "C:/Users/DELL/Desktop/brain-tumor-detection-master/finaldata/"
        img = cv2.imread(os.path.join(path, 'contour.png'))
        grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        
        dilate = cv2.imread(os.path.join(path, "contour.png"))
        img3 = cv2.applyColorMap(dilate, cv2.COLORMAP_HOT)

        cv2.imwrite(os.path.join(path, 'coloring.png'), img3)

        pixmap = QtGui.QPixmap(os.path.join(path, 'coloring.png'))  # Setup pixmap with the provided image
        pixmap = pixmap.scaled(self.label_2.width(), self.label_2.height(),
                               QtCore.Qt.KeepAspectRatio)  # Scale pixmap
        self.label_2.setPixmap(pixmap)  # Set the pixmap onto the label

    def saving(self):
        import cv2
        import os
        from PyQt5 import QtWidgets
        import glob, os, os.path

        path="C:/Users/DELL/Desktop/brain-tumor-detection-master/"

        img = cv2.imread(os.path.join(path, 'coloring.png'))

        fname,_ = QtWidgets.QFileDialog.getSaveFileName(None, 'Open file',"C:/Users/DELL/Desktop/brain-tumor-detection-master/"
                                                        , "Image files (*.jpg)")

        if fname:
            cv2.imwrite(fname, img)
            error_dialog = QtWidgets.QErrorMessage()
            error_dialog.setWindowTitle('Image processing tool')
            error_dialog.setWindowIcon(QIcon("C:/Users/DELL/Desktop/brain-tumor-detection-master/finaldata/"))
            error_dialog.showMessage('Image Saved')
            error_dialog.exec()

            self.label_2.clear()
            self.label.clear()
            filelist=glob.glob(os.path.join(path,"*.png"))
            for f in filelist:
                os.remove(f)
                print('deleted')

        else:
            print('notsaved')
            error_dialog = QtWidgets.QErrorMessage()
            error_dialog.setWindowTitle('Image processing tool')
            error_dialog.setWindowIcon(QIcon('C:/Users/DELL/Desktop/brain-tumor-detection-master/finaldata/'))
            error_dialog.showMessage('Please save image!')
            error_dialog.exec()
            self.label_2.clear()
            self.label.clear()
            filelist = glob.glob(os.path.join(path, "*.png"))
            for f in filelist:
                os.remove(f)


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.showMaximized()
    MainWindow.show()
    sys.exit(app.exec_())

    








