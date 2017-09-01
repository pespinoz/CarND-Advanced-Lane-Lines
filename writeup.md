
## **Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[im1]: ./output_images/cam_calibration_images/comparisons/calibration3.jpg "Undistorted"
[im2]: ./output_images/undistorted_images/comparisons/test4.jpg "Undistorted"
[im3]: ./output_images/gradient_thresholded_images/test2_resized.jpg "Gradient Binary"
[im4]: ./output_images/color_thresholded_images/test2_resized.jpg "Color Binary"
[im5]: ./output_images/combined_threshold_images/comparisons/test3.jpg "Final Binary"
[im6]: ./output_images/perspective_transformation_images/final_straight_warped.jpg "Perspective Transformation"
[im7]: ./output_images/warped_images/comparisons/test2.jpg "Warped"
[im8]: ./output_images/lane_detection_images/detection_test2.jpg "Detection"
[im9]: ./examples/color_fit_lines.jpg "Fit Example"
[im10]: ./output_images/lane_detection_images/final_test2_resized.jpg "Unwarped"
[vi1]: ./project_video_output.mp4 "Video"

### Files Included:
My project includes the following files in the top level directory, "./":
* `cam_calibration.py:` Computes distortion coefficients and camera matrix.
* `perspective_transformation.py:` Computes the transform matrix **M**.
* `helper_functions.py:` This file contains a several functions that aid on the implementation of the video pipeline.  
* `video_pipeline.py:` This file contains the implementation of the video pipeline.
* `writeup.md:` This file.
* `project_video_output.mp4:` A video successfully showing the lane lines detection.

And the folders:
* `output_images/`: Here I store the images produced by the _single images pipeline_.
* `pickle_files/`: Stores distortion coefficients and transform matrix that are applied to all images (and frames) in this project.
* `single_images_pipeline/`:  Includes (in execution order)
    - `undistortion.py`
    - `gradient_threshold.py` 
    - `color_threshold.py`
    - `combined_threshold.py` 
    - `warp.py`
    - `lane_detection.py`

---

### Running the Code:

The video pipeline is implemented in `./video_pipeline.py`. This generates our [output mp4 file][vi1]. All the functions used in this pipeline are implemented in the file `helper_functions.py`. 

In `./single_images_pipeline/*.py`, I tweaked parameters and produced plots to assure the pipeline worked on (single) test images. I took advantage of this knowledge to eventually build the video pipeline. 

---

### [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the file `cam_calibration.py` located in "./", the top-level directory of the project. Running this script results in obtaining the camera calibration and distortion coefficients. Both of these are saved in the file `camera_calibration.pickle`  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image. Corner detection is achieved with the `findChessboardCorners()` function, assuming a 9x6 board. `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to all the calibration images in `camera_cal/*` using the `cv2.undistort()` function. An example of this process is shown in the following Figure: 

![alt text][im1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

The code for this step is contained in the file `undistortion.py` located in "./single_images_pipeline". Running this script results in obtaining the undistorted version of the test images included in the project. To undistort images I load the distortion coefficients from the `camera_calibration.pickle` file and the `cv2.undistort()` function.  

To demonstrate this step, see the Figure below. In the left side we have the original test image, and in the right side we show its undistorted version. Note the white car to make the differences between these images evident:

![alt text][im2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The code for this step is contained in the files `gradient_threshold.py` and `color_threshold.py` located in "./single_images_pipeline". Running these scripts results in obtaining thresholded binary images for each of the following:

`gradient_threshold.py`: We try to identify lane lines by applying:
* Both an horizontal and vertical Sobel operator, with thresholds in a range of (50, 255), and kernel size 5.
* Repeat the previous step, with the same parameters of threshold and kernel, but this time computing the magnitude of the gradient.
* Then we compute the direction of the gradient, with a threshold in a range of (0.5, 1.3) in radians.

Next we combine these gradient-criteria to obtain a unique gradient-thresholded binary image. An example is shown in the next Figure:

![alt text][im3]

`color_threshold.py`: We convert the original test images to the RGB, HLS, YCrCb, and LAB colorspaces. After visual inspection and trial and error, I selected the R, S, Cr, and L channels as useful in the task of detecting lane lines for changing conditions of both pavement color and shadows in the images. We applied thresholds of:
* (217, 255) in the R channel image.
* (109, 255) in the S channel image. 
* (140, 255) in the Cr channel image.
* (227, 255) in the L channel image.

Next we combine these color-criteria to obtain a unique color-thresholded binary image. An example is shown in the Figure below:

![alt text][im4]


Finally I used a combination of the color and gradient thresholds developed above to generate a final binary image. This is done in the file `combined_threshold.py`.  Here's an example of my output for this last thresholding step, next to the original test image for comparison purposes:

![alt text][im5]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

Having a formulation to obtain binary-thresholded images (described above), then we must warp them to obtain a _bird's eye_ perspective.

The code for my perspective transform is given in the file `perspective_transformation.py` located in "./single_images_pipeline". Running this script results in obtaining the transform matrix **M**, with the aid of the `getPerspectiveTransform()` function. **M** is saved in the file `transformation.pickle`. Once I obtain the transform matrix, I can apply it to each of the test images (or video frames), to obtain the _bird's eye_ perspective. 
 
In `perspective_transformation.py` we derive different transforms as function of the parameter **cte** (see code below). We take as inputs an image (undistorted straight lines), as well as source (`src`) and destination (`dst`) points.  I chose to hardcode the source and destination points in the following manner:

```python
offsetx = 100
offsety = 0

roi1 = [[-cte, IM_SIZE[1]], [int(IM_SIZE[0] / 2 - 70), 446], [int(IM_SIZE[0] / 2) + 70, 446],
        [IM_SIZE[0] + cte, IM_SIZE[1]]]
roi2 = [[offsetx, IM_SIZE[1] - offsety], [offsetx, offsety], [IM_SIZE[0] - offsetx, offsety],
        [IM_SIZE[0] - offsetx, IM_SIZE[1] - offsety]]

src = np.float32(roi1)
dst = np.float32(roi2)
```

The parameter **cte** is optimized by the visual inspection of almost a hundred warped images. We set **cte=160**, as this results in the lane lines in the warped images to appear the "most" parallel. Now we can give numerical values for both the source and destination points below:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| -160, 720     | 100, 720      | 
|  570, 446     | 100,   0      |
|  710, 446     | 1180,  0      |
| 1440, 720     | 1180, 720     |

Once again, I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto both straight-line test images and their warped counterparts to verify that the lines appear parallel in the warped images.

![alt text][im6]

Since the transform matrix **M** works, I can apply it now to all of my test images. I do this with the script `warp.py` located in "./single_images_pipeline". An example is shown in the image below.

![alt text][im7]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Our goal is to identify lane-line pixels in a thresholded(binary)-warped image as the one shown above. Once detected, we can fit their positions with a 2nd order polynomial.

So, for the lane-line pixels identification part, we start by taking a histogram along the columns in the lower half of the image. The two peaks in this histogram are  good indicators of the x-position of the base of the lane lines. This is a starting point for where to search: Next we can use a sliding window, to find and follow the lines up to the top of the image. This sliding window procedure is implemented within the `sliding_windows()` function, defined in `line 160` of `helper_functions.py`. This function is used in the `lane_detection.py` file (located in "./single_images_pipeline"), where I visualize the results for the test images set:

![alt text][im8]

Once the lane lines pixels are identified we fit 2nd order polynomials to the left and right lanes. A good illustrative scheme of this procedure is given by the next Figure (taken from the Udacity lesson):

![alt text][im9]

I defined the best-fit function in `second_order_polyfit()` (in `line 265` of `helper_functions.py`). This function is used in the `lane_detection.py` file (located in "./single_images_pipeline"), where I derive best-fits for the test images set. 

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Having the best-fit to the identified lane-lines from the previous step, we can calculate both the radius of curvature and offset of the vehicle from the center. To this end, we use the formula for the radius of curvature at any point _y_ for a curve _x=f(y)_ introduced [here](http://www.intmath.com/applications-differentiation/8-radius-curvature.php), and defined mathematically [here](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/2b62a1c3-e151-4a0e-b6b6-e424fa46ceab/lessons/40ec78ee-fb7c-4b53-94a8-028c5c60b858/concepts/2f928913-21f6-4611-9055-01744acc344f). The same link applies for the offset from the center formula. 

In `lane_detection.py` we use the functions `get_curvature()`, and `get_offset()`. Both of these are defined in lines 276 and 292 of `helper_functions.py` respectively.

The results for my test images are shown in the next Table. For the conversion between image and physical units, we assumed 30 meters per 720 pixels in the y dimension, and 3.7 meters per 700 meters in the x dimension.

| Image         | Curvature (meters)    |  Offset (meters) |   
|:-------------:|:---------------------:|:----------------:| 
| test1           |  2510 m    | 0.212 m| 
| test2           |  408 m     | 0.386 m|
| test3           |  406 m     | 0.179 m|
| test4           |  585 m     | 0.287 m|
| test5           |  464 m     | 0.058 m|
| test6           |  1338 m    | 0.246 m|
| straight_lines1 |  147098 m  | 0.058 m|
| straight_lines2 |  3126 m    | 0.083 m|

Note that the curvature values in the Table are the average between the left- and right-line best fits.


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The final step of my (single images) pipeline is to project our left and right lane best-fits back onto the road. I implemented this step in lines 299 and 314 of `helper_functions.py`, with the functions `draw_fit_in_image()` and `generate_annotated_final()`. 

Here I'm starting from a _bird's view_ of a binary image, where I have polynomial fits to the left and right lanes, and doing a perspective transform back to the original camera view. To do this I use the inverse of **M**, the perspective transform matrix introduced above. The space between the lines is filled with a green polygon. Finally we annotate the numeric values for the radius of curvature and lane position offset onto the image as well.

I applied this procedure to the test images in lines 51 and 53 of `lane_detection.py`. Here is one example of my results:

![alt text][im10]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result][vi1].

The video pipeline is implemented only in one file: `video_pipeline.py`, located in the top-level directory of the project.

In "./single_images_pipeline", I tweaked parameters and produced plots to assure the pipeline worked on (single) test images. Then I took advantage of this knowledge and built a modularized video pipeline in `video_pipeline.py`. All the functions used in the video pipeline are implemented in the file `helper_functions.py`. 

In `video_pipeline.py` I introduce the class `Line()` to receive  characteristics of each line detection and store them. Tracking them allows me to tell the frames where detection is problematic for any reason. This distinction is made within the `sanity_check()` function (line 80). When that happens, I just revert to the last _bona fide_ detection. Also, the curvature of radius and offset from center annotated in the images are smoothed over the _n_ last _bona fide_ frame detections (lines 74-77 of `video_pipeline.py`).

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

My approach was to start with a single-images pipeline and then generalize into a video one. In the former I tweaked parameters and tried several threshold-combination approaches; then applied this knowledge to the latter pipeline.

When I implemented the video pipeline, I also added a class that tracks lane detection in each frame and returns _False_ when a detection is problematic for any reason. In those cases I just revert to the last _bona fide_ detection. This improvement greatly enhances the stability of the video output, especially in some difficult passages (which include pavement color changes, or shadows). In general, the pipeline performs flawlessly in the project video. Likewise, i) the curvature radius, and ii) offset from the center, show numerical values that make sense based on where the measurements were taken. 

Unfortunately the pipeline (at least with the present parameters) does not perform well on the challenge videos. These show different roads, with different pavement conditions. Also in the "harder_challenge_video.mp4" the turns have much lower curvature radii, as well as significant brightness variations and even dash reflections that make lane detection very difficult. Many of the quantities we have used as parameters in this project (hard coded regions, thresholds, etc) become obsolete when environmental and driving conditions are changed. In that sense, it is possible that an approach based on convolutional neural networks (used in P3), but now trained for lane line detection could prove useful to generalize our results.
