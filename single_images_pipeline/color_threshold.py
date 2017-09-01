import matplotlib.pyplot as plt
from helper_functions import *

images, filenames = load_images_as_rgb(UNDIST_IMS)      # read undistorted images and its filenames # PIPELINE!
create_or_rewrite(COLOR_THRESH_IMS)                     # dir for color-thresholded images
create_or_rewrite(COLOR_THRESH_IMS + 'comparisons')     # dir for color-thresholded ims comparison to originals

########################################################################################################################

color_binary = []
for fname, img in zip(filenames, images):
    r_channel = rgb_select(img, 1, thresh=(221, 255))
    s_channel = hls_select(img, 3, thresh=(109, 255))
    cr_channel = ycrcb_select(img, 2, thresh=(140, 255))
    l_channel = lab_select(img, 1, thresh=(227, 255))

    combination = np.zeros_like(s_channel)
    combination[((cr_channel == 1) | (l_channel == 1)) & (s_channel == 1) | (r_channel == 1)] = 1
    color_binary.append(combination)
    cv2.imshow('color threshold image', 255*combination)
    cv2.waitKey(500)
    cv2.imwrite(COLOR_THRESH_IMS + fname, 255*combination)
cv2.destroyAllWindows()

########################################################################################################################

# Visualize thresholding and save it for a list of given images
givens = [0, 1, 2, 3, 4, 5, 6, 7]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=SCREENSIZE)
for given in givens:
    ax1.imshow(images[given])
    ax1.set_title('Original Image: ' + filenames[given], fontsize=15)
    ax2.imshow(color_binary[given], cmap='gray')
    ax2.set_title('Color Threshold (combined): ' + filenames[given], fontsize=15)
    fig.savefig(COLOR_THRESH_IMS + 'comparisons/' + filenames[given], dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None, transparent=False, bbox_inches='tight',
                pad_inches=0.1, frameon=None)
