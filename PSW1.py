
# python -m pip  install -U  matplotlib
import numpy as np
import matplotlib.pyplot as plt


resolution = 32
depth = 2

foo = np.linspace(0,4*np.pi,resolution)

print(foo)


bar = np.zeros((16,16))
bar[3:6,2:9] = 1

print(bar)

fix, ax = plt.subplots(2,3, figsize=(
    10,5
))

# print(ax)

ax[0][0].plot(foo, c='red', ls=":", lw=4, label='CZERWONA LINIA')
# some rectangle using ones in bar
# ax[1][0].imshow(bar, cmap='binary')


# plot value vector of sinus sampled with foo
reflectance = np.sin(foo)
ax[0][1].plot(reflectance)

#sinus - plot [0][1] od [0][0] (albo odwrotnie?)
ax[0][2].plot(foo, reflectance)


ax[0][1].grid(ls=":")
ax[0][1].spines['top'].set_visible(False)
ax[0][1].spines['right'].set_visible(False)
ax[0][1].set_xlabel('os X')
ax[0][1].set_ylabel('os Y')
ax[0][0].legend()


# mape cieplna maciezy 2 wym, ktora jest iloczynem 2
image = reflectance[:,np.newaxis] * reflectance[np.newaxis,:]
ax[1][0].imshow(image, cmap='binary', vmin=0, vmax=1)

#normalizacja przedzialowa - dodac jeden i podzielic przez 2
# nrange = (-2,2)
nrange = (-1,1)
# nrange = (-.001,.001)
norm_image = (image - nrange[0]) / (nrange[1]-nrange[0])
norm_image = np.clip(norm_image, 0, 1)
ax[1][1].imshow(norm_image, cmap='binary', vmin=0, vmax=1)

# pokazanie w 4bitowej glebi - piksel zapisany od 0 do 15
dmin, dmax = (0, np.power(2, depth)-1)
digital_image = np.rint(norm_image*dmax)
ax[1][2].imshow(digital_image, cmap='binary', vmin=dmin, vmax=dmax)

for i in range(3):
    ax[1][i].get_xaxis().set_visible(False)
    ax[1][i].get_yaxis().set_visible(False)

plt.tight_layout()
plt.savefig('foo.png')
