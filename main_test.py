from myfoobar import int_img, Harr1, Harr2, Harr3, Harr4, testingERM
import numpy as np
import os
import matplotlib.image as mpimg
# -------------------------------------------------------------------------------------
img_dir_faces = 'C:/Users/lijin/Desktop/Fall 2020/ECEN649/Project/trainset/faces/'
img_dir_non_faces = 'C:/Users/lijin/Desktop/Fall 2020/ECEN649/Project/testset/non-faces/'
numofAdaRound = 10
overallthreshold = 0
jstar_all = np.load('jstar_all.npy')
thetastar_all = np.load('thetastar_all.npy')
wt = np.load('wt.npy')
Harr_map = np.load('HarrMap.npy')
# -------------------------------------------------------------------------------------
print('Calculating accuracy of faces ......')
numoffaces = len([lists for lists in os.listdir(img_dir_faces) if os.path.isfile(os.path.join(img_dir_faces, lists))])
data_path = os.listdir(img_dir_faces)
label_all_faces = np.ones(numoffaces, dtype=int)
label_all_predict_faces = []

for f1 in data_path:
    img_path = img_dir_faces + f1
    img = mpimg.imread(img_path)
    imglength = len(img)
    integral_image = img.copy()
    int_img(integral_image, imglength)
    WL = 0
    for T in range(numofAdaRound):
        location = Harr_map[jstar_all[T]]
        harrtype = location[0]
        if harrtype == 1:
            feature = Harr1(integral_image, location)
        elif harrtype == 2:
            feature = Harr2(integral_image, location)
        elif harrtype == 3:
            feature = Harr3(integral_image, location)
        elif harrtype == 4:
            feature = Harr4(integral_image, location)
        WL += wt[T] * testingERM(feature, thetastar_all[T])
    label_all_predict_faces.append(np.sign(WL - overallthreshold))
# print('label_all_predict faces=', label_all_predict_faces)
accuracy_faces = sum(label_all_predict_faces == label_all_faces) / numoffaces
print('accuracy of faces= ', accuracy_faces)
# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
print('Calculating accuracy of non faces ......')
data_path2 = os.listdir(img_dir_non_faces)
numofnonfaces = len([lists for lists in os.listdir(img_dir_non_faces) if os.path.isfile(os.path.join(img_dir_non_faces, lists))])
label_all_nonfaces = -1 * np.ones(numofnonfaces, dtype=int)
label_all_predict_nonfaces = []
for f2 in data_path2:
    img_path = img_dir_non_faces + f2
    img = mpimg.imread(img_path)
    imglength = len(img)
    integral_image = img.copy()
    int_img(integral_image, imglength)
    WL = 0
    for T in range(numofAdaRound):
        location = Harr_map[jstar_all[T]]
        harrtype = location[0]
        if harrtype == 1:
            feature = Harr1(integral_image, location)
        elif harrtype == 2:
            feature = Harr2(integral_image, location)
        elif harrtype == 3:
            feature = Harr3(integral_image, location)
        elif harrtype == 4:
            feature = Harr4(integral_image, location)
        WL += wt[T] * testingERM(feature, thetastar_all[T])
    label_all_predict_nonfaces.append(np.sign(WL - overallthreshold))
# print('label_all_predict nonfaces =', label_all_predict_nonfaces)
accuracy_nonfaces = sum(label_all_predict_nonfaces == label_all_nonfaces) / numofnonfaces
print('accuracy of nonfaces = ', accuracy_nonfaces)