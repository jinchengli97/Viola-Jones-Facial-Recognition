from myfoobar import ERMdecisionstumps, testERM
import numpy as np
import os

img_dir_faces = 'C:/Users/lijin/Desktop/Fall 2020/ECEN649/Project/trainset/faces/'
img_dir_non_faces = 'C:/Users/lijin/Desktop/Fall 2020/ECEN649/Project/trainset/non-faces/'
trainingset = np.load('trainingset.npy')
numoffaces = len([f for f in os.listdir(img_dir_faces) if os.path.isfile(os.path.join(img_dir_faces, f))])
numofnonfaces = len([f for f in os.listdir(img_dir_non_faces) if os.path.isfile(os.path.join(img_dir_non_faces, f))])
Distribution = np.array([0.5/numoffaces] * numoffaces + [0.5/numofnonfaces] * numofnonfaces)
# Distribution = np.array([1/row] * row)

row, column = len(trainingset), len(trainingset[0])
print('row (number of sample) =', row)
print('column (number of Harr Feature + label) =', column)
wt, jstar_all, thetastar_all = [], [], []

for T in range(10):
    # Calculate threshold and polarization
    print('AdaBoost progress at %dth round: ' % (T+1))
    Distribution = Distribution / np.sum(Distribution)
    jstar, thetastar = ERMdecisionstumps(trainingset, Distribution)
    jstar_all.append(jstar)
    thetastar_all.append(thetastar)

    # compute accuracy & print accuracy
    label = testERM(trainingset, jstar, thetastar)
    y = trainingset[:, -1]
    accuracy = sum(y == label) / row
    print('accuracy of the %d th round is %f.' % (T+1, accuracy))

    # Compute weight and adjust distribution
    idx = np.where(y != label)
    Epsilon = np.sum(Distribution[idx])
    wt.append(0.5 * np.log(1/Epsilon - 1))

    # Update Di, in t+1 th round
    denominator = np.sum(Distribution * np.exp(-wt[T] * y * label))
    for i in range(row):
        Distribution[i] = Distribution[i] * np.exp(-wt[T] * y[i] * label[i]) / denominator
    print('jstar =', jstar, 'thetastar =', thetastar, 'wt =', wt)


# Save important parameters
np.save('jstar_all.npy', np.array(jstar_all))
np.save('thetastar_all.npy', np.array(thetastar_all))
np.save('wt.npy', np.array(wt))