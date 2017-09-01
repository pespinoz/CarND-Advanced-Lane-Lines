from numpy.linalg import inv
import matplotlib.pyplot as plt
import pickle
from helper_functions import *

with open('../pickle_files/transformation.pickle', 'rb') as f:
    M = pickle.load(f)

undists, _ = load_images_as_rgb(UNDIST_IMS)             # read undistorted images
images, filenames = load_images_as_rgb(WARPED_IMS)      # read warped-binary images and its filenames

create_or_rewrite(DETECT_IMS)

########################################################################################################################

givens = [0, 1, 2, 3, 4, 5, 6, 7]

for choice in givens:
    image = images[choice][:, :, 0]
    filename = filenames[choice]

    histogram = np.sum(image[image.shape[0]//2:, :], axis=0)

    out, left_x_pos, left_y_pos, right_x_pos, right_y_pos, left_inds, right_inds, y, x = \
        sliding_windows(image, 8, 100, 50)

    yy, left_fit, right_fit = second_order_polyfit(left_x_pos, left_y_pos, right_x_pos, right_y_pos)

    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=SCREENSIZE)
    out[y[left_inds], x[left_inds]] = [255, 0, 0]
    out[y[right_inds], x[right_inds]] = [0, 0, 255]
    ax1.imshow(out)
    ax1.plot(left_fit, yy, color='yellow')
    ax1.plot(right_fit, yy, color='yellow')
    ax2.imshow(image)
    fig2.savefig(DETECT_IMS + 'detection_' + filename, dpi=None, facecolor='w', edgecolor='w',
                 orientation='portrait', papertype=None, format=None, transparent=False, bbox_inches='tight',
                 pad_inches=0.1, frameon=None)

########################################################################################################################

    _, _, left_curvature_m, right_curvature_m = get_curvatures(yy, left_fit, right_fit)
    print(filename)
    print(left_curvature_m, 'm', right_curvature_m, 'm')   # Now our radius of curvature is in meters
    offset = get_offset(yy, left_fit, right_fit)
    print(offset, 'm')


########################################################################################################################

    im_with_fit = draw_fit_in_image(image, yy, left_fit, right_fit)

    final = generate_annotated_final(im_with_fit, undists[choice], inv(M),
                                     (right_curvature_m + left_curvature_m)/2, offset)

    fig3, ax1 = plt.subplots(1, 1, figsize=SCREENSIZE)
    result = cv2.addWeighted(undists[choice], 1, final, 0.7, 0)
    ax1.imshow(result)
    fig3.savefig(DETECT_IMS + 'final_' + filename, dpi=None, facecolor='w', edgecolor='w',
                 orientation='portrait', papertype=None, format=None, transparent=False, bbox_inches='tight',
                 pad_inches=0.1, frameon=None)
