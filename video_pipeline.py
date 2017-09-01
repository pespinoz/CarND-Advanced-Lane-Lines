from moviepy.editor import VideoFileClip
from numpy.linalg import inv
import pickle
from helper_functions import *

# single images pipeline: run cam_calibration.py, undistortion.py, gradient_threshold.py, color_threshold.py
# combined_threshold.py, warp.py and lane_detection.py.

# video pipeline: run cam_calibration.py, perspective_transformation.py, and main.py


class Line(object):
    # Define a class to receive the characteristics of each line detection
    def __init__(self):
        self.detected = []              # was the line detected in the last iteration?
        self.leftlane_fit = []          # polynomial fit x coordinates to the left lane
        self.rightlane_fit = []         # polynomial fit x coordinates to the right lane
        self.radius_of_curvature = []   # radius of curvature
        self.offset_from_center = []    # offset from the center
        self.input_video_clip = VideoFileClip(video_input)
        self.output_video_clip = self.input_video_clip.fl(self.pipeline)

    def pipeline(self, gf, t):
        # define my video pipeline based on the results from my single images pipeline:
        img = gf(t)
        undistorted = undistort(img, mtx, dist, None, mtx)

        # magnitude and color thresholds:
        gradx_binary = abs_sobel_threshold(undistorted, orient='x', sobel_kernel=ksize, thresh=grad_thresh)
        grady_binary = abs_sobel_threshold(undistorted, orient='y', sobel_kernel=ksize, thresh=grad_thresh)
        mag_binary = mag_threshold(undistorted, sobel_kernel=ksize, thresh=mag_thresh)
        dir_binary = dir_threshold(undistorted, sobel_kernel=ksize, thresh=dir_thresh)
        r_channel = rgb_select(undistorted, 1, thresh=r_thresh)
        s_channel = hls_select(undistorted, 3, thresh=s_thresh)
        cr_channel = ycrcb_select(undistorted, 2, thresh=cr_thresh)
        l_channel = lab_select(undistorted, 1, thresh=l_thresh)

        # combination of criteria:
        thresholded = np.zeros_like(dir_binary)
        thresholded[(((gradx_binary == 1) & (grady_binary == 1)) | ((mag_binary == 1) & (dir_binary == 1))) |
                    (((cr_channel == 1) | (l_channel == 1)) & (s_channel == 1) | (r_channel == 1))] = 1

        # bird's eye perspective:
        warped = cv2.warpPerspective(255 * thresholded, M, IM_SIZE)
        warped[(warped < 100)], warped[(warped >= 100)] = 0, 255

        # lane detection:
        _, left_x_pos, left_y_pos, right_x_pos, right_y_pos, left_inds, right_inds, y, x = sliding_windows(warped,
                                                                                                           nwindows,
                                                                                                           margin,
                                                                                                           minpix)

        # lane polynomial fit:
        yy, left_fit, right_fit = second_order_polyfit(left_x_pos, left_y_pos, right_x_pos, right_y_pos)
        # curvature radius:
        _, _, left_curvature_m, right_curvature_m = get_curvatures(yy, left_fit, right_fit)
        # offset from center
        offset = get_offset(yy, left_fit, right_fit)
        # tracking characteristics:
        self.radius_of_curvature.append((left_curvature_m + right_curvature_m) / 2)
        self.offset_from_center.append(offset)
        self.leftlane_fit.append(left_fit)
        self.rightlane_fit.append(right_fit)
        self.detected.append(self.sanity_check(left_fit, right_fit))

        # if not detected in the last frame, look for the last detection:
        if self.detected[-1]:
            warped_with_fit = draw_fit_in_image(warped, yy, self.leftlane_fit[-1], self.rightlane_fit[-1])
        else:
            ind = np.where(np.array(self.detected) == True)[0][-1]
            warped_with_fit = draw_fit_in_image(warped, yy, self.leftlane_fit[ind], self.rightlane_fit[ind])

        # smooth the curvature of radius and offset from center annotated in the images:
        inds = list(np.where(np.array(self.detected) == True)[0][-average_n:])
        output = generate_annotated_final(warped_with_fit, undistorted, inv(M),
                                          np.mean([self.radius_of_curvature[v] for v in inds]),
                                          np.mean([self.offset_from_center[v] for v in inds]))
        return output

    def sanity_check(self, leftfit, rightfit):
        # defines if the lanes were detected in the last frame or not:
        dist_up = (leftfit[0] - rightfit[0]) * XM_PER_PIX
        dist_bottom = (leftfit[-1] - rightfit[-1]) * XM_PER_PIX
        std = np.std(leftfit - np.mean(leftfit) - (rightfit - np.mean(rightfit)))
        if std >= 50 or dist_bottom <= -3.5 or dist_up <= -4.1:
            return False
        else:
            return True

########################################################################################
########################################################################################

with open('pickle_files/camera_calibration.pickle', 'rb') as f:
    ret, mtx, dist, rvecs, tvecs = pickle.load(f)
with open('pickle_files/transformation.pickle', 'rb') as f:
    M = pickle.load(f)

ksize = 5  # Choose a Sobel kernel size, a larger odd number gives smoother gradient measurements
grad_thresh = (50, 255)  # thresholds (color and magnitude) derived from the single (images) pipeline.
mag_thresh = (50, 255)
dir_thresh = (0.5, 1.3)
r_thresh = (217, 255)
s_thresh = (109, 255)
cr_thresh = (140, 255)
l_thresh = (227, 255)
nwindows = 25  # Choose the number of sliding windows
margin = 100  # Set the width of the windows +/- margin
minpix = 50  # Set minimum number of pixels found to recenter window

average_n = 12
video_input = './project_video.mp4'
video_output = './project_video_output.mp4'

result = Line()
result.output_video_clip.write_videofile(video_output, audio=False)
