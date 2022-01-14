# # python -m pip  install -U  matplotlib
import numpy as np
import matplotlib.pyplot as plt

resolution = 256
depth = 2
drange = (-1,1)
n_iter = 256
N = np.power(2, depth)-1

prober = np.linspace(0,8*np.pi,resolution)
prober = np.sin(prober)
perfect_image = prober[:,np.newaxis] * prober[np.newaxis,:]

shape = len(perfect_image)

n_matrix = np.zeros((shape,shape))
o_matrix = np.zeros((shape,shape))

fix, ax = plt.subplots(2,3, figsize=(
    12,8
))

for i in range(2):
    for j in range(3):
        ax[i][j].get_xaxis().set_visible(False)
        ax[i][j].get_yaxis().set_visible(False)

for i in range(n_iter):
    noise = np.random.normal(0,1,size=(shape, shape))
    n_image = perfect_image + noise
    o_image = perfect_image

    n_dimg = (n_image - drange[0]) / (drange[1]-drange[0])
    n_dimg = np.clip(n_dimg, 0, 1)
    n_dimg = np.rint(n_dimg*N)

    o_dimg = (o_image - drange[0]) / (drange[1]-drange[0])
    o_dimg = np.clip(o_dimg, 0, 1)
    o_dimg = np.rint(o_dimg*N)

    n_matrix = n_matrix + n_dimg
    o_matrix = o_matrix + o_dimg

    ax[0][0].imshow(perfect_image,cmap='binary')
    ax[1][0].imshow(noise,cmap='binary')
    ax[0][1].imshow(o_dimg,cmap='binary', vmin=0, vmax=N)
    ax[1][1].imshow(n_dimg,cmap='binary', vmin=0, vmax=N)
    ax[0][2].imshow(o_matrix,cmap='binary')
    ax[1][2].imshow(n_matrix,cmap='binary')
    plt.tight_layout()
    plt.savefig('foo.png')


# przyczyna różnic pomiędzy wynikowymi obrazami akwizycji z kolumny trzeciej.
# obraz z binarnego wejscia stal sie 8 bitowy
# szum ma wieksza glebie niz pierwotny obraz i
# dolny jest odszumiany - z kazda klatka wylania sie coraz wyrazniejszy obraz
