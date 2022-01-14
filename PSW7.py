import numpy as np
import matplotlib.pyplot as plt
# from numpy import random
# from skimage import morphology
from skimage import draw, transform
# from skimage.transform import rotate
from scipy.signal import convolve2d
# from scipy.signal import medfilt
#alt+shift+e to executed marked code
# https://gist.github.com/xehivs/a176f923fffed0a371e3b5d319797dc1

np.random.seed(0)

entry_img = np.zeros(shape=(256, 256))

rectangles = [
    draw.rectangle(start=(32, 32), end=(92, 92)),
    draw.rectangle(start=(92, 92), end=(128, 128)),
    draw.rectangle(start=(128, 128), end=(144, 144)),
]

for rect in rectangles:
    entry_img[tuple(rect)] = 1

angle = np.random.uniform(-20, 20)
rotated_img = transform.rotate(entry_img, angle)

s1 = [[-1,0,1], [-2,0,2], [-1,0,1]]
s3 = [[1,2,1], [0,0,0], [-1,-2,-1]]
angles = np.linspace(-30, 30, 50)
abssums_s1 = np.zeros(shape=(50,))
abssums_s2 = np.zeros(shape=(50,))



for i in range(50):
    fig, axs = plt.subplots(3, 2)

    angle = angles[i]
    out_img = transform.rotate(rotated_img, angle)
    conv_s1_img = convolve2d(out_img, s1)
    conv_s3_img = convolve2d(out_img, s3)

    abssums_s1[i] = np.abs(conv_s1_img).sum()
    abssums_s2[i] = np.abs(conv_s3_img).sum()


    axs[0][0].imshow(entry_img, cmap='binary')
    axs[0][1].imshow(rotated_img,  cmap='binary')
    axs[1][0].imshow(out_img,  cmap='binary')
    axs[1][1].imshow(conv_s1_img,  cmap='binary')
    axs[2][1].imshow(conv_s3_img,  cmap='binary')
    axs[2][0].plot(angles, abssums_s1)
    axs[2][0].plot(angles, abssums_s2)


    # plt.show()
    plt.savefig("frame.png")
    plt.close()

min_s1 = np.argmin(abssums_s1)
min_s3 = np.argmin(abssums_s2)
best_angle1 = angles[min_s1]
best_angle2 = angles[min_s3]
best_angle = np.mean((angles[min_s1],angles[min_s3]))

final_img = transform.rotate(rotated_img, best_angle)
plt.imsave("baz.png", final_img)

color_img = np.stack((entry_img,final_img, rotated_img), axis=2)
plt.imsave("bar.png", color_img)

# biały - część wspólna
#niebieski - obraz zniekształcony o losowy kąt
# żółty - różnica miedzy wejsciowym a koncowym
# czerwony i zielony- wejsciowy i koncowy