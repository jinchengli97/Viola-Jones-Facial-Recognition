import numpy as np
imglength = 19
map1, map2, map3, map4 = [], [], [], []

# Harr feature 1 construction
for length in range(4, 10, 1):              # vertical length 4,5,6,7,8,9
    for width in range(2, 10, 2):           # horizontal length 2, 4, 6, 8
        for i in range(1, imglength - length):
            for j in range(1, imglength - width):
                map1.append([1, i, j, i + length - 1, j + width - 1])

# Harr feature 2 construction
for length in range(2, 10, 2):              # vertical length 2,4,6,8 has to be divided by 2
    for width in range(4, 16, 1):           # horizontal length 4,5,6....,16
        for i in range(1, imglength - length):
            for j in range(1, imglength - width):
                map2.append([2, i, j, i + length - 1, j + width - 1])

# Harr feature 3 construction
for length in range(4, 12, 1):              # vertical length 6,7,8,9,10,11,12
    for width in range(6, 15, 3):           # horizontal length 6,9,12has to be divided by 3
        for i in range(1, imglength - length):
            for j in range(1, imglength - width):
                map3.append([3, i, j, i + length - 1, j + width - 1])

# Harr feature 4 construction
for length in range(4, 12, 2):              # vertical length 4,6,8,10 has to be divided by 2
    for width in range(4, 12, 2):           # horizontal length 4,6,8,10 has to be divided by 2
        for i in range(1, imglength - length):
            for j in range(1, imglength - width):
                map4.append([4, i, j, i + length - 1, j + width - 1])
HarrMap = map1 + map2 + map3 + map4
HarrMap = np.array(HarrMap)
np.save('Harr1Map.npy', map1)
np.save('Harr2Map.npy', map2)
np.save('Harr3Map.npy', map3)
np.save('Harr4Map.npy', map4)
np.save('HarrMap.npy', HarrMap)

