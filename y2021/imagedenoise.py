import matplotlib.pyplot as plt

from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)
from skimage import data, img_as_float
from skimage.util import random_noise
import cv2


original = cv2.imread("BlankHXL.png")
original = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)

sigma = 0.155
noisy = random_noise(original, var=sigma**2)

fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(8, 5),
                       sharex=True, sharey=True)

plt.gray()

# Estimate the average noise standard deviation across color channels.
sigma_est = estimate_sigma(noisy, average_sigmas=True)
# Due to clipping in random_noise, the estimate will be a bit smaller than the
# specified sigma.
print(f'Estimated Gaussian noise standard deviation = {sigma_est}')

ax[0, 0].imshow(noisy)
ax[0, 0].axis('off')
ax[0, 0].set_title('Noisy')
print("a")
ax[0, 1].imshow(denoise_tv_chambolle(noisy, weight=0.1))
ax[0, 1].axis('off')
ax[0, 1].set_title('TV')
ax[0, 2].imshow(denoise_bilateral(noisy, sigma_color=0.05, sigma_spatial=15))
ax[0, 2].axis('off')
print("b")

ax[0, 2].set_title('Bilateral')
ax[0, 3].imshow(denoise_wavelet(noisy, rescale_sigma=True))
ax[0, 3].axis('off')
print("c")

ax[0, 3].set_title('Wavelet denoising')

ax[1, 1].imshow(denoise_tv_chambolle(noisy, weight=0.2))
ax[1, 1].axis('off')
ax[1, 1].set_title('(more) TV')
print("d")

ax[1, 2].imshow(denoise_bilateral(noisy, sigma_color=0.1, sigma_spatial=15))
ax[1, 2].axis('off')
ax[1, 2].set_title('(more) Bilateral')
ax[1, 3].imshow(denoise_wavelet(noisy, convert2ycbcr=True,
                                rescale_sigma=True))
print("f")

ax[1, 3].axis('off')
ax[1, 3].set_title('Wavelet denoising\nin YCbCr colorspace')
ax[1, 0].imshow(original)
ax[1, 0].axis('off')
ax[1, 0].set_title('Original')

fig.tight_layout()

plt.show()