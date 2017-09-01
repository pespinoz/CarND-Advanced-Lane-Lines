import pickle
from helper_functions import *
import matplotlib.pyplot as plt

images, filenames = load_images_as_rgb(CALIB_IMS)   # read calibration images and its filenames
create_or_rewrite(CORNER_IMS)                       # create directory for calibration images marked with corners
create_or_rewrite(CORNER_IMS + 'comparisons')

########################################################################################################################

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((CHESS_Y*CHESS_X, 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESS_X, 0:CHESS_Y].T.reshape(-1, 2)

objpoints = []  # 3d points in real world space, array to store object points
imgpoints = []  # 2d points in image plane, array to store image points
for fname, img in zip(filenames, images):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Find the chessboard corners (son pares ordenados en calib im)
    ret, corners = cv2.findChessboardCorners(gray, (CHESS_X, CHESS_Y), None)
    # If found, add object points, image points
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (CHESS_X, CHESS_Y), corners, ret)
        cv2.imshow('img', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(500)
        cv2.imwrite(CORNER_IMS + fname, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    else:
        print(fname)
cv2.destroyAllWindows()
# you should now have `objpoints` and `imgpoints` needed for camera calibration.

# Do camera calibration given object points and image points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, IM_SIZE, None, None)

givens = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=SCREENSIZE)
for given in givens:
    ax1.imshow(images[given])
    ax1.set_title('Original Image: ' + filenames[given], fontsize=15)
    ax2.imshow(undistort(images[given], mtx, dist, None, mtx))
    ax2.set_title('Undistorted Image: ' + filenames[given], fontsize=15)
    fig.savefig(CORNER_IMS + 'comparisons/' + filenames[given], dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None, transparent=False, bbox_inches='tight',
                pad_inches=0.1, frameon=None)

########################################################################################################################

# Save the camera calibration result for later use
with open('pickle_files/camera_calibration.pickle', 'wb') as f:
    pickle.dump([ret, mtx, dist, rvecs, tvecs], f)
