import os
import cv2
import shutil
import numpy as np

SCREENSIZE = (16, 8.5)
IM_SIZE = (1280, 720)
VARS_PATH = 'output_images/'

CALIB_IMS = 'camera_cal/'
CORNER_IMS = 'output_images/cam_calibration_images/'
CHESS_X = 9
CHESS_Y = 6
TRANS_IMS = 'output_images/perspective_transformation_images/'
STRAIGHT_IMS = ('output_images/undistorted_images/straight_lines1.jpg',
                'output_images/undistorted_images/straight_lines2.jpg')

TEST_IMS = '../test_images/'
UNDIST_IMS = '../output_images/undistorted_images/'

GRAD_THRESH_IMS = '../output_images/gradient_thresholded_images/'
COLOR_THRESH_IMS = '../output_images/color_thresholded_images/'
THRESH_IMS = '../output_images/combined_threshold_images/'

WARPED_IMS = '../output_images/warped_images/'

YM_PER_PIX = 30/720  # meters per pixel in y dimension
XM_PER_PIX = 3.7/700  # meters per pixel in x dimension
DETECT_IMS = '../output_images/lane_detection_images/'


def load_images_as_rgb(folder):
    loaded_images, loaded_filenames = [], []
    for filename in os.listdir(folder):
        if os.path.isfile(os.path.join(folder, filename)):
            im = cv2.imread(os.path.join(folder, filename))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            if im is not None:
                loaded_images.append(im)
                loaded_filenames.append(filename)
    print("Loaded {} images".format(len(loaded_images)))
    return loaded_images, loaded_filenames


def create_or_rewrite(folder_name):
    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)
    os.makedirs(folder_name)


def undistort(source, cam_mtx, dist_coeffs, output, new_cam_mtx):
    dst = cv2.undistort(src=source, cameraMatrix=cam_mtx, distCoeffs=dist_coeffs, dst=output,
                        newCameraMatrix=new_cam_mtx)
    return dst


def abs_sobel_threshold(img, orient, sobel_kernel, thresh):
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y',
    # 3) Take the absolute value of the derivative or gradient
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude is > thresh_min and < thresh_max
    binary = np.zeros_like(scaled_sobel)
    binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary


def mag_threshold(img, sobel_kernel, thresh):
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Calculate the magnitude
    gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scale_factor = np.max(gradmag) / 255
    gradmag = (gradmag / scale_factor).astype(np.uint8)
    # 5) Create a binary mask where mag thresholds are met
    binary = np.zeros_like(gradmag)
    binary[(gradmag >= thresh[0]) & (gradmag <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary


def dir_threshold(img, sobel_kernel, thresh):
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    # 5) Create a binary mask where direction thresholds are met
    binary = np.zeros_like(absgraddir)
    binary[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    # Return the binary image
    return binary


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    # Draws `lines` with `color` and `thickness`. Lines are drawn on the image inplace (mutates the image).
    # For semi-transparent lines, think about combining this function with the weighted_img() function below
    for line in lines:
        for x1, y1, x2, y2 in line:
            x1 = round(x1)
            x2 = round(x2)
            y1 = round(y1)
            y2 = round(y2)
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def rgb_select(img, ch, thresh=(0, 255)):
    # 1) Convert to HLS color space
    channel = img[:, :, ch-1]
    # 2) Apply a threshold to the S channel
    binary_output = np.zeros_like(channel)
    binary_output[(channel > thresh[0]) & (channel <= thresh[1])] = 1
    return binary_output


def hls_select(img, ch, thresh=(0, 255)):
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    channel = hls[:, :, ch-1]
    # 2) Apply a threshold to the S channel
    binary_output = np.zeros_like(channel)
    binary_output[(channel > thresh[0]) & (channel <= thresh[1])] = 1
    return binary_output


def ycrcb_select(img, ch, thresh=(0, 255)):
    # 1) Convert to HLS color space
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    channel = ycrcb[:, :, ch-1]
    # 2) Apply a threshold to the S channel
    binary_output = np.zeros_like(channel)
    binary_output[(channel > thresh[0]) & (channel <= thresh[1])] = 1
    return binary_output


def lab_select(img, ch, thresh=(0, 255)):
    # 1) Convert to HLS color space
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    channel = lab[:, :, ch-1]
    # 2) Apply a threshold to the S channel
    binary_output = np.zeros_like(channel)
    binary_output[(channel > thresh[0]) & (channel <= thresh[1])] = 1
    return binary_output


def sliding_windows(binary_warped, nwindows, margin, minpix):
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255  # output im to draw on and see the result

    midpoint = np.int(histogram.shape[0]/2)        # Find the peak of the left and right halves of the histogram
    leftx_base = np.argmax(histogram[:midpoint])   # These will be the starting point for the left and right lines
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    window_height = np.int(binary_warped.shape[0]/nwindows)     # Set height of windows

    nonzero = binary_warped.nonzero()       # Identify the x and y indices of all nonzero pixels in the image
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx_current = leftx_base              # Current positions to be updated for each window
    rightx_current = rightx_base

    left_lane_inds = []                     # Create empty lists to receive left and right lane pixel indices
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) &
                          (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) &
                           (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx, lefty = nonzerox[left_lane_inds], nonzeroy[left_lane_inds]
    rightx, righty = nonzerox[right_lane_inds], nonzeroy[right_lane_inds]

    return out_img, leftx, lefty, rightx, righty, left_lane_inds, right_lane_inds, nonzeroy, nonzerox


def window_mask(width, height, img_ref, center, level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output


def find_window_centroids(image, window_width, window_height, margin):
    window_centroids = []  # Store the (left,right) window centroid positions per level
    window = np.ones(window_width)  # Create our window template that we will use for convolutions

    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template

    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(image[int(3 * image.shape[0] / 4):, :int(image.shape[1] / 2)], axis=0)
    l_center = np.argmax(np.convolve(window, l_sum)) - window_width / 2
    r_sum = np.sum(image[int(3 * image.shape[0] / 4):, int(image.shape[1] / 2):], axis=0)
    r_center = np.argmax(np.convolve(window, r_sum)) - window_width / 2 + int(image.shape[1] / 2)

    # Add what we found for the first layer
    window_centroids.append((l_center, r_center))

    # Go through each layer looking for max pixel locations
    for level in range(1, (int)(image.shape[0] / window_height)):
        # convolve the window into the vertical slice of the image
        image_layer = np.sum(
            image[int(image.shape[0] - (level + 1) * window_height):int(image.shape[0] - level * window_height), :],
            axis=0)
        conv_signal = np.convolve(window, image_layer)
        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
        offset = window_width / 2
        l_min_index = int(max(l_center + offset - margin, 0))
        l_max_index = int(min(l_center + offset + margin, image.shape[1]))
        l_center = np.argmax(conv_signal[l_min_index:l_max_index]) + l_min_index - offset
        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center + offset - margin, 0))
        r_max_index = int(min(r_center + offset + margin, image.shape[1]))
        r_center = np.argmax(conv_signal[r_min_index:r_max_index]) + r_min_index - offset
        # Add what we found for that layer
        window_centroids.append((l_center, r_center))

    y = np.linspace((int)(window_height / 2), (int)(image.shape[0] - (window_height / 2)),
                    (int)(image.shape[0] / window_height))
    return window_centroids, list(np.array(window_centroids)[:].T[0]), list(y)[::-1], \
           list(np.array(window_centroids)[:].T[1]), list(y)[::-1]


def second_order_polyfit(left_x, left_y, right_x, right_y):
    # Fit a second order polynomial to each
    left_fit = np.polyfit(left_y, left_x, 2)
    right_fit = np.polyfit(right_y, right_x, 2)

    ploty = np.linspace(0, IM_SIZE[1]-1, IM_SIZE[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    return ploty, left_fitx, right_fitx


def get_curvatures(ploty, left_fitx, right_fitx):
    y_eval = np.max(ploty)
    # The formula for curvature
    left_curv = ((1 + (2*left_fitx[0]*y_eval + left_fitx[1])**2)**1.5) / np.absolute(2*left_fitx[0])
    right_curv = ((1 + (2*right_fitx[0]*y_eval + right_fitx[1])**2)**1.5) / np.absolute(2*right_fitx[0])

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*YM_PER_PIX, left_fitx*XM_PER_PIX, 2)
    right_fit_cr = np.polyfit(ploty*YM_PER_PIX, right_fitx*XM_PER_PIX, 2)

    # Calculate the new radii of curvature
    left_curv_m = ((1 + (2*left_fit_cr[0]*y_eval*YM_PER_PIX + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curv_m = ((1 + (2*right_fit_cr[0]*y_eval*YM_PER_PIX + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    return left_curv, right_curv, left_curv_m, right_curv_m


def get_offset(ploty, left_fitx, right_fitx):
    # You can assume the camera is mounted at the center of the car, such that the lane center is the midpoint at the
    # bottom of the image between the two lines you detected. The offset of the lane center from the center of the image
    # (converted from pixels to meters) is your distance from the center of the lane.
    return ((right_fitx[len(ploty)-1] + left_fitx[len(ploty)-1]) / 2 - (IM_SIZE[0] / 2)) * XM_PER_PIX


def draw_fit_in_image(binary_warped, ploty, left_fitx, right_fitx):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    return color_warp


def generate_annotated_final(im_with_fit, undist, Minv, curvature, offset):
    curvature = int(curvature)
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    unwarp = cv2.warpPerspective(im_with_fit, Minv, IM_SIZE)
    # Combine the result with the original image
    output = cv2.addWeighted(undist, 1, unwarp, 0.3, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(output, 'Radius of Curvature = {:d} (m)'.format(curvature),
                (45, 60), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(output, 'Lane Position (offset) = {:.2} (m)'.format(offset),
                (45, 155), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
    return output
