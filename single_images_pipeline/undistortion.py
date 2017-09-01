import pickle
import matplotlib.pyplot as plt
from helper_functions import *

with open('../pickle_files/camera_calibration.pickle', 'rb') as f:
    ret, mtx, dist, rvecs, tvecs = pickle.load(f)

images, filenames = load_images_as_rgb(TEST_IMS)        # read test images and its filenames # PIPELINE!
create_or_rewrite(UNDIST_IMS)                           # dir for undistorted images
create_or_rewrite(UNDIST_IMS + 'comparisons')           # dir for undistorted-images comparison to originals

########################################################################################################################

undistorted = []
for fname, img in zip(filenames, images):
    out = undistort(img, mtx, dist, None, mtx)
    undistorted.append(out)
    cv2.imshow('undistorted image', cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    cv2.waitKey(500)
    cv2.imwrite(UNDIST_IMS + fname, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
cv2.destroyAllWindows()

########################################################################################################################

# Visualize undistortion and save undistortion comparison for a a list of given images
givens = [0, 1, 2, 3, 4, 5, 6, 7]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=SCREENSIZE)
for given in givens:
    ax1.imshow(images[given])
    ax1.set_title('Original Image: ' + filenames[given], fontsize=15)
    ax2.imshow(undistorted[given])
    ax2.set_title('Undistorted Image: ' + filenames[given], fontsize=15)
    fig.savefig(UNDIST_IMS + 'comparisons/' + filenames[given], dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None, transparent=False, bbox_inches='tight',
                pad_inches=0.1, frameon=None)
