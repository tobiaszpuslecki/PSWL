import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from skimage import morphology
#alt+shift+e to executed marked code

shape = (32,64)

rawImg = np.zeros(shape=shape)

numberOfPoints = 256


while (numberOfPoints):
    x = random.randint(0,shape[0])
    y = random.randint(0,shape[1])
    if rawImg[x][y] == 1:
        continue
    else:
        rawImg[x][y] = 1
        numberOfPoints = numberOfPoints-1;

        if x < shape[0]-1:
            rawImg[x-1][y] = 1
        if y < shape[1]-1:
            rawImg[x][y+1] = 1




fig, axs = plt.subplots(3, 2)

axs[0][0].imshow(rawImg, cmap='binary')

#
selen = morphology.disk(1)
axs[0][1].imshow(selen)

eorsion = morphology.erosion(rawImg, selen)
axs[1][1].imshow(eorsion)

dilatation = morphology.dilation(rawImg, selen)
axs[1][0].imshow(dilatation)

opening = morphology.opening(rawImg, selen)
axs[2][0].imshow(opening)


closing = morphology.closing(rawImg, selen)
axs[2][1].imshow(closing)



plt.show()
