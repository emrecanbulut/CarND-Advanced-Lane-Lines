**Advanced Lane Finding Project**

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

[image1]: ./test_images_output/undistort_compared.png "Undistorted"
[image2]: ./test_images_output/undistorted_output.png "Road Transformed"
[image3]: ./test_images_output/combined_binary_output.png "Binary Example"
[image4]: ./test_images_output/warped_straight_lines.png "Warp Example"
[image5]: ./test_images_output/colored_lane_marked.png "Fit Visual"
[image6]: ./test_images_output/example_output.png "Output"
[video1]: ./output_videos/project_video.mp4 "Video" 

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.
The code for this part can be found in `calibrateMyCamera()` method. First of all, I replicated some points of the chessboard 
that represents the real world space. Then, I read all the calibration images and converted them to `GRAY` 
before using the OpenCV method `findChessboardCorners(gray, (9,6),None)`. After each image where successfully found the corners, I added 
replicated points to `objpoints` list and the corners found to `imgpoints` list. Afterwards, I used 
`calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)` of OpenCV.

![Undistorted comparison][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![Undistorted road][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used color transforms and gradient in combination to create a binary image. This part can be found in `colorAndGradient()` method. 
I converted the color space into HLS and excluded the H channel. I used the L channel to take the derivative in x (sobel x) and I used 
the S channel for color thresholds. Then combined the output of them in one frame. The output looks like this:
![Combined binary output][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for this part can be found in `perspectiveTransform()` method. The method takes an image as input. I chose source(`src`) and 
destination(`dst`) points. Then, I used `getPerspectiveTransform(src, dst)` of OpenCV to find the `transformationMatrix` and warp the image 
using `warpPerspective` method of OpenCV.

```python

src = np.float32([
	(imshape[0]*0.156, imshape[1]),
	(imshape[0]*0.469, imshape[1]*0.62), 
	(imshape[0]*0.529, imshape[1]*0.62), 
	(imshape[0]*0.862, imshape[1])])
dst = np.float32([
	(imshape[0]*0.3, imshape[1]*0.98),
	(imshape[0]*0.3, imshape[1]*0.01),
	(imshape[0]*0.7, imshape[1]*0.01),
	(imshape[0]*0.7, imshape[1]*0.98)])

```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 199.68, 720   | 384, 705.6    | 
| 600.32, 446.4 | 384, 7.2      |
| 677.12, 446.4 | 896, 7.2      |
| 1103.36, 720  | 896, 705.6    |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![Warped image][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The part where I identified lane-line pixels can be found in `find_lane_pixels(binary_warped)`. Shortly, in the `binary_warped` input 
image, I vertically checked where most of the nonzero values lied, and added their indices to lane-line pixels list.

The part where I fit a 2nd order polynomial to the lane-line pixels can be found in `fit_poly()` method.

![Poly fitted][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I calculated the radius of curvature in `calculateCurvature()` method. And the vehicle position calculation was based on the assumption that 
the camera was mounted exactly in the middle of the car. So, the pixel difference between midpoint of the lane and the midpoint of the frame 
multiplied by pixel-per-kilometer constant results in the position offset.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in `transformBackToOriginal` method using the inverse of `transformationMatrix`
![Example output frame][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_videos/project_video.mp4)


### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?
I haven't worked on the sanity of the lane lines that were detected. Therefore, once it detects some other line as the lane line, it will keep searching the lane line around the wrong line and 
will almost always fail to find the correct one.