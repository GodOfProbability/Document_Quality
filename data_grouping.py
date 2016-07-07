import numpy as np
#import matplotlib
import matplotlib.pyplot as plt
from scipy import signal
from scipy import misc
from scipy.signal import convolve2d
#from skimage import data
from skimage.morphology import disk
from skimage.filters import threshold_otsu, rank
from skimage.util import img_as_ubyte
import h5py


def std_convoluted(image, N):
    im = np.array(image, dtype=float)
    im2 = im ** 2
    ones = np.ones(im.shape)

    kernel = np.ones((2 * N + 1, 2 * N + 1))
    s = convolve2d(im, kernel, mode="same")
    s2 = convolve2d(im2, kernel, mode="same")
    ns = convolve2d(ones, kernel, mode="same")
    temp = np.absolute((s2 - s ** 2 / 25) / 25)
    # mask = np.isreal(temp)
    # print (np.count_nonzero(mask))
    return np.sqrt(temp)


def normalize(x):
    mean_ker = np.ones((5, 5)) / 25
    mean = signal.convolve2d(x, mean_ker, boundary='symm', mode='same')
    std = std_convoluted(x, 2)
    blurr_image = (x - mean) / (std + 1e-5)
    return blurr_image


def preprocess(x):
    img = x.astype(np.uint8)
    img = img_as_ubyte(img)

    threshold_global_otsu = threshold_otsu(img)
    global_otsu = img >= threshold_global_otsu
    mask = np.ones_like(img, dtype=np.bool_)
    rows, cols = img.shape
    for i in range(0, rows, 48):
        for j in range(0, cols, 48):
            patch = global_otsu[i:i + 48, j:j + 48]
            summ = np.sum(patch)
            if summ == 0 or summ == 48 * 48:
                mask[i:i + 48, j:j + 48] = 0
    return mask


def patches(mask):
    ir = range(170, 3050)
    ic = range(120, 1650)
    selected_indices = []
    count = 0
    rand_ir = np.random.choice(ir, (100), replace=False)
    rand_ic = np.random.choice(ic, (100), replace=False)
    for r in rand_ir:
        for c in rand_ic:
            summ = np.sum(mask[r:r + 48, c:c + 48]) > 48 * 24
            if summ:
                selected_indices += [(r, c)]
                count += 1
                if count == 850:
                    return selected_indices
    return selected_indices


path = '/data4/gopal.sharma/Image_Quality/'

scores_file = open(path + 'myfile', 'r')
listing = []
for scores in scores_file:
    listing += [scores]
groups = open('indices.txt', 'r')
count = 0
for i, f in enumerate(groups):
    print ('The value of f: ', f)
    lower, upper = f.split(',')
    lower = int(lower)
    upper = int(upper)
    size = range(lower, upper + 1)
    X = np.zeros((850 * len(size), 48, 48))
    Y = np.zeros((850 * len(size)))
    for gr_i, gr_v in enumerate(size):
        x = plt.imread(path + 'images/%d.jpg' % (gr_v))
        x = np.mean(x, axis=2)
        blurr_image = normalize(x)  # Normalize using 5 X 5 % kernel
        mask = preprocess(x)  # Do the Otsu binrization
        selected_indices = patches(mask)  # Extract indices of patches from the masking
        for indices, value in enumerate(selected_indices):
            r, c = value
            X[850 * gr_i + indices, :, :] = blurr_image[r:r + 48, c:c + 48]
            Y[850 * gr_i + indices] = float(listing[gr_v - 1])
        count += 1
        print ('value of count: ', count)
    with h5py.File('grouping%d.hdf5' % (i), 'w') as hf:
        hf.create_dataset('X', data=X)
        hf.create_dataset('y', data=Y)




