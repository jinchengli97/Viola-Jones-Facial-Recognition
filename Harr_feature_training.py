from myfoobar import int_img, Harr1, Harr2, Harr3, Harr4
import matplotlib.image as mpimg
import numpy as np
import os
import time

img_dir_faces = 'C:/Users/lijin/Desktop/Fall 2020/ECEN649/Project/trainset/faces/'
img_dir_non_faces = 'C:/Users/lijin/Desktop/Fall 2020/ECEN649/Project/trainset/non-faces/'
Harr1Map = np.load('Harr1Map.npy')
Harr2Map = np.load('Harr2Map.npy')
Harr3Map = np.load('Harr3Map.npy')
Harr4Map = np.load('Harr4Map.npy')
HarrMap = np.load('HarrMap.npy')

# ------------------------------------------
tic = time.time()
# ------------------------------------------
# training faces
data_path = os.listdir(img_dir_faces)
trainingset = []
for f1 in data_path:
    HF = []
    img_path = img_dir_faces + f1
    img = mpimg.imread(img_path)
    imglength = len(img)
    integral_image = img.copy()
    int_img(integral_image, imglength)
    for location in Harr1Map:
        HF.append(Harr1(integral_image, location))
    for location in Harr2Map:
        HF.append(Harr2(integral_image, location))
    for location in Harr3Map:
        HF.append(Harr3(integral_image, location))
    for location in Harr4Map:
        HF.append(Harr4(integral_image, location))
    label = [1]
    feature = HF + label
    trainingset.append(feature)
# ------------------------------------------
# training non faces
data_path = os.listdir(img_dir_non_faces)
for f1 in data_path:
    HF = []
    img_path = img_dir_non_faces + f1
    img = mpimg.imread(img_path)
    imglength = len(img)
    integral_image = img.copy()
    int_img(integral_image, imglength)
    for location in Harr1Map:
        HF.append(Harr1(integral_image, location))
    for location in Harr2Map:
        HF.append(Harr2(integral_image, location))
    for location in Harr3Map:
        HF.append(Harr3(integral_image, location))
    for location in Harr4Map:
        HF.append(Harr4(integral_image, location))
    label = [-1]
    feature = HF + label
    trainingset.append(feature)
# ------------------------------------------
toc = time.time()
# ------------------------------------------
print('total time = ', toc-tic, 's')
trainingset = np.array(trainingset)
np.save('trainingset.npy', trainingset)

