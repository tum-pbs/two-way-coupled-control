import matplotlib.pyplot as plt
import numpy as np
import os


def plot(*args, scalex=True, scaley=True, data=None, block=False, **kwargs):
    # plt.close('all')
    plt.figure()
    plt.plot(*args, scalex=True, scaley=True, data=None, **kwargs)
    plt.show(block=block)


def imshow(X, cmap=None, norm=None, aspect=None, interpolation=None, alpha=None, vmin=None, vmax=None, origin='lower', extent=None, save=False, show=True, name=None, block=False, **kwargs):
    # plt.close('all')
    plt.figure()
    if X.ndim != 2: X = X[0, ..., 0]
    plt.imshow(X, cmap, norm, aspect, interpolation, alpha, vmin, vmax, origin, extent, **kwargs)
    plt.colorbar()

    if save == True:
        dirName = './debug'
        if not os.path.isdir(dirName): os.mkdir(dirName)

        if name == None:
            isSaveNotSucessful = True
            i = -1
            while isSaveNotSucessful:
                i += 1
                filePath = dirName + '/debug%d.png' % i
                if not os.path.isfile(filePath):
                    plt.savefig(filePath, dpi=100)
                    isSaveNotSucessful = False

        else: plt.savefig("%s/%s.png" % (dirName, name), dpi=100)
    if show == True: plt.show(block=block)
