import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import os
from moviepy.editor import VideoFileClip


def calibrateMyCamera():
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob('camera_cal/calibration*.jpg')

    # Step through the list and search for chessboard corners
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return ret, mtx, dist, rvecs, tvecs

def colorAndGradient(img, s_thresh=(90, 180), sx_thresh=(20, 100)):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    #green: gradient (sxbinary),  blue: color threshold
    #color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255

    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    return combined_binary

def readAndCvt(fname):
    img = cv2.imread(fname)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def showBeforeAndAfter(before_img, after_img):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(before_img)#, cmap='gray')
    ax1.set_title('Before Image', fontsize=50)
    ax2.imshow(after_img)#, cmap='gray')
    ax2.set_title('After Image', fontsize=50)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    plt.show()
    #plt.savefig('test_images_output\warped_straight_lines')

def drawPolygon(img):
    output = np.copy(img)
    imshape = output.shape[1::-1]
    vertices = np.array([(imshape[0]*0.156,imshape[1]),(imshape[0]*0.469, imshape[1]*0.62), (imshape[0]*0.529, imshape[1]*0.62), (imshape[0]*0.862,imshape[1])], dtype=np.int32)
    vertices = vertices.reshape((-1,1,2))
    cv2.polylines(output, [vertices], True, 1)
    return output

def perspectiveTransform(img):
    global transMatrix
    imshape = img.shape[1::-1]
    src = np.float32([(imshape[0]*0.156,imshape[1]),(imshape[0]*0.469, imshape[1]*0.62), (imshape[0]*0.529, imshape[1]*0.62), (imshape[0]*0.862,imshape[1])])
    dst = np.float32([(imshape[0]*0.3,imshape[1]*0.98),(imshape[0]*0.3, imshape[1]*0.01), (imshape[0]*0.7, imshape[1]*0.01), (imshape[0]*0.7,imshape[1]*0.98)])
    #dst = np.float32([(imshape[0]*0.3,imshape[1]),(imshape[0]*0.1, 0), (imshape[0]*0.9, 0), (imshape[0]*0.7,imshape[1])])
    # Given src and dst points, calculate the perspective transform matrix
    transMatrix = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    unwarped = cv2.warpPerspective(img, transMatrix, imshape, flags=cv2.INTER_LINEAR)
    return unwarped

def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    # margin to be ignored on the horizontal
    ignored_margin = int(histogram.shape[0]*0.15)
    midpoint = np.int(histogram.shape[0]//2)

    leftx_base = np.argmax(histogram[ignored_margin:midpoint]) + ignored_margin
    rightx_base = np.argmax(histogram[midpoint:(histogram.shape[0]-ignored_margin)]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 100

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
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
        
        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img

def fit_poly(img_shape, leftx, lefty, rightx, righty):
     ### TO-DO: Fit a second order polynomial to each with np.polyfit() ###
    global lane_mid_point
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    ### TO-DO: Calc both polynomials using ploty, left_fit and right_fit ###
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    lane_mid_point = (left_fitx[img_shape[0]-1] + right_fitx[img_shape[0]-1])/2
    return left_fitx, right_fitx, ploty, left_fit, right_fit

def search_around_poly(binary_warped):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 50

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    ### TO-DO: Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    ### Hint: consider the window areas for the similarly named variables ###
    ### in the previous quiz, but change the windows to our new search area ###
    global prev_left_fit, prev_right_fit, left_curverad, right_curverad

    left_lane_inds = ((nonzerox > (prev_left_fit[0]*(nonzeroy**2) + prev_left_fit[1]*nonzeroy + 
                    prev_left_fit[2] - margin)) & (nonzerox < (prev_left_fit[0]*(nonzeroy**2) + 
                    prev_left_fit[1]*nonzeroy + prev_left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (prev_right_fit[0]*(nonzeroy**2) + prev_right_fit[1]*nonzeroy + 
                    prev_right_fit[2] - margin)) & (nonzerox < (prev_right_fit[0]*(nonzeroy**2) + 
                    prev_right_fit[1]*nonzeroy + prev_right_fit[2] + margin)))
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
    # update previous fits for the next iteration
    left_fitx, right_fitx, ploty, prev_left_fit, prev_right_fit = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)
    y_eval = np.max(ploty)
    left_curverad, right_curverad = calculateCurvature(prev_left_fit.copy(), prev_right_fit.copy(), y_eval)

    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()

    left_line_window = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    right_line_window = np.array([np.flipud(np.transpose(np.vstack([right_fitx, 
                              ploty])))])
    lane_pts = np.hstack((left_line_window, right_line_window))
    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([lane_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    
    # Plot the polynomial lines onto the image
    '''
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    '''
    ## End visualization steps ##   
    return result, window_img

def cvtParabola(left_fit_cr, right_fit_cr):
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/500 # meters per pixel in x dimension
    left_fit_cr[0] *= xm_per_pix / (ym_per_pix ** 2)
    left_fit_cr[1] *= xm_per_pix / ym_per_pix
    right_fit_cr[0] *= xm_per_pix / (ym_per_pix ** 2)
    right_fit_cr[1] *= xm_per_pix / ym_per_pix
    return left_fit_cr, right_fit_cr

def calculateCurvature(left_fit_cr, right_fit_cr, y_eval):
    left_fit_cr, right_fit_cr = cvtParabola(left_fit_cr, right_fit_cr)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    return int(left_curverad), int(right_curverad)
    
def work_on_video(filename='project_video.mp4'):
    white_output = 'output_videos/'+ filename
    ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
    ## To do so add .subclip(start_second,end_second) to the end of the line below
    ## Where start_second and end_second are integer values representing the start and end of the subclip
    ## You may also uncomment the following line for a subclip of the first 5 seconds
    ##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
    clip1 = VideoFileClip(filename)#.subclip(0,10) #solidWhiteRight.mp4")
    white_clip = clip1.fl_image(video_processor) #NOTE: this function expects color images!!
    white_clip.write_videofile(white_output, audio=False)

def video_processor(image):
    processed_img, undist = process_image(image)
    unwarped = transformBackToOriginal(processed_img)
    result = cv2.addWeighted(undist, 1, unwarped, 0.8, 0)
    cv2.putText(
        img=result,
        text='Curvature radius: '+str((right_curverad+left_curverad)/2),
        org=(0, 50),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=(255, 255, 255))
    
    offset = vehiclePosition(lane_mid_point)
    if offset < 0:
        offset = -offset
        msg = 'm right of the lane center'
    else:
        msg = 'm left of the lane center'
    msg = str(offset)+msg

    cv2.putText(
        img=result,
        text='Vehicle position: '+msg,
        org=(0, 100),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=(255, 255, 255))
    return result

def vehiclePosition(lane_midpoint_x):
    xm_per_pixel = 3.7 / 500.0 
    frame_midpoint_x = 639 #(1280/2)
    pixel_difference = lane_midpoint_x - frame_midpoint_x
    return pixel_difference*xm_per_pixel

def work_on_images():
    global isFirstFrame
    images = glob.glob('test_images/*.jpg')
    for file in images:
        isFirstFrame = True
        image = readAndCvt(file)
        processed_img, undist = process_image(image)
        unwarped = transformBackToOriginal(processed_img)
        result = cv2.addWeighted(undist, 1, unwarped, 0.8, 0)
        #showBeforeAndAfter(processed_img, result)

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    global mtx, dist, isFirstFrame, prev_left_fit, prev_right_fit
    undist = cv2.undistort(image, mtx, dist, None, mtx)
    combined_binary = colorAndGradient(undist)
    #saveImage(combined_binary, 'Combined Binary Image', 'combined_binary_output')
    #roi_drawn = drawPolygon(undist)
    #roi_drawn = cv2.cvtColor(roi_drawn,cv2.COLOR_BGR2RGB)
    binary_warped = perspectiveTransform(combined_binary)
    #showBeforeAndAfter(roi_drawn, binary_warped)
    if(isFirstFrame):
        leftx, lefty, rightx, righty, _ = find_lane_pixels(binary_warped)
        _, _, _, prev_left_fit, prev_right_fit = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)
        isFirstFrame = False
    out_img, _= search_around_poly(binary_warped)
    return out_img, undist

def saveImage(img, title, filename):
    cv2.imwrite('test_images_output/'+filename+'.png',img)
    '''
    plt.imshow(img)# cmap='gray')
    plt.title(title)    
    if not os.path.exists('test_images_output'):
        os.makedirs('test_images_output')
    plt.savefig('test_images_output/'+filename)
    '''

def transformBackToOriginal(processed_img):
    global transMatrix
    invTransMatrix = np.linalg.inv(transMatrix)
    imshape = processed_img.shape[1::-1]
    warped = cv2.warpPerspective(processed_img, invTransMatrix, imshape, flags=cv2.INTER_LINEAR)
    return warped

ret, mtx, dist, rvecs, tvecs = calibrateMyCamera()

transMatrix = None
isFirstFrame = True
prev_left_fit = None
prev_right_fit = None
left_curverad = None
right_curverad = None
vehicle_pos = None
lane_mid_point = None

if __name__ == '__main__':
    work_on_video()
    #work_on_images()
