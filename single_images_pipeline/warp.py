import matplotlib.pyplot as plt
import pickle
from helper_functions import *

with open('../pickle_files/transformation.pickle', 'rb') as f:
    M = pickle.load(f)

images, filenames = load_images_as_rgb(THRESH_IMS)      # read inter-thresholded-images and its filenames
create_or_rewrite(WARPED_IMS)                           # dir for unwarp images
create_or_rewrite(WARPED_IMS + 'comparisons')           # dir for unwarp-images comparison to originals

########################################################################################################################

warpeds = []
for fname, img in zip(filenames, images):
    warped = cv2.warpPerspective(img, M, IM_SIZE)
    warped[(warped < 100)], warped[(warped >= 100)] = 0, 255
    warpeds.append(warped)

    cv2.imshow('warped image', warped)
    cv2.waitKey(500)
    cv2.imwrite(WARPED_IMS + fname, warped)
cv2.destroyAllWindows()

########################################################################################################################

# Visualize unwarping and save it for a list of given images
givens = [0, 1, 2, 3, 4, 5, 6, 7]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=SCREENSIZE)
for given in givens:
    ax1.imshow(images[given])
    ax1.set_title('Undistorted/Thresholded Image: ' + filenames[given], fontsize=15)
    ax2.imshow(warpeds[given])
    ax2.set_title('Warped: ' + filenames[given], fontsize=15)
    fig.savefig(WARPED_IMS + 'comparisons/' + filenames[given], dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None, transparent=False, bbox_inches='tight',
                pad_inches=0.1, frameon=None)
