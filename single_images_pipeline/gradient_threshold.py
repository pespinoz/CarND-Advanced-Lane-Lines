import matplotlib.pyplot as plt
from helper_functions import *

images, filenames = load_images_as_rgb(UNDIST_IMS)      # read undistorted images and its filenames # PIPELINE!
create_or_rewrite(GRAD_THRESH_IMS)                      # dir for gradient-thresholded images
create_or_rewrite(GRAD_THRESH_IMS + 'comparisons')      # dir for gradient-thresholded ims comparison to originals

# Parameters
ksize = 5  # Choose a Sobel kernel size, a larger odd number gives smoother gradient measurements
grad_thresh = (50, 255)
mag_thresh = (50, 255)
dir_thresh = (0.5, 1.3)

########################################################################################################################

gradient_binary = []
for fname, img in zip(filenames, images):
    gradx_binary = abs_sobel_threshold(img, orient='x', sobel_kernel=ksize, thresh=grad_thresh)
    grady_binary = abs_sobel_threshold(img, orient='y', sobel_kernel=ksize, thresh=grad_thresh)
    mag_binary = mag_threshold(img, sobel_kernel=ksize, thresh=mag_thresh)
    dir_binary = dir_threshold(img, sobel_kernel=ksize, thresh=dir_thresh)

    combination = np.zeros_like(dir_binary)
    combination[((gradx_binary == 1) & (grady_binary == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    gradient_binary.append(combination)
    cv2.imshow('gradient threshold image', 255*combination)
    cv2.waitKey(500)
    cv2.imwrite(GRAD_THRESH_IMS + fname, 255*combination)
cv2.destroyAllWindows()

########################################################################################################################

# Visualize gradient thresholding and save it for a list of given images
givens = [0, 1, 2, 3, 4, 5, 6, 7]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=SCREENSIZE)
for given in givens:
    ax1.imshow(images[given])
    ax1.set_title('Original Image: ' + filenames[given], fontsize=15)
    ax2.imshow(gradient_binary[given], cmap='gray')
    ax2.set_title('Gradient Threshold (combined): ' + filenames[given], fontsize=15)
    fig.savefig(GRAD_THRESH_IMS + 'comparisons/' + filenames[given], dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None, transparent=False, bbox_inches='tight',
                pad_inches=0.1, frameon=None)
