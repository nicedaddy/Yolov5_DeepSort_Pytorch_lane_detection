#!/usr/bin/env python

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import glob
import os
import sys

### STEP 1: Camera Calibration ###

def distortion_factors():
    # Prepare object points
    # From the provided calibration images, 9*6 corners are identified 
    nx = 11 #9 , 11
    ny = 8 #6 , 8
    objpoints = []
    imgpoints = []
    # Object points are real world points, here a 3D coordinates matrix is generated
    # z coordinates are 0 and x, y are equidistant as it is known that the chessboard is made of identical squares
    # objp = np.zeros((6*9,3), np.float32)
    objp = np.zeros((8*11,3), np.float32)
    # objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
    objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)
  
    # Make a list of calibration images
    os.listdir("camera_cal/")
    cal_img_list = os.listdir("camera_cal/")  
    
    # Imagepoints are the coresspondant object points with their coordinates in the distorted image
    # They are found in the image using the Open CV 'findChessboardCorners' function
    for image_name in cal_img_list:
        import_from = 'camera_cal/' + image_name
        img = cv2.imread(import_from)
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        # If found, draw corners
        if ret == True:
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (nx, ny), corners, ret)#
            imgpoints.append(corners)
            objpoints.append(objp)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    ###################################
    ## checking the undistored image ##
    ###################################
    # for img_name in cal_img_list:
    #     import_from = 'camera_cal/' + img_name
    #     img = cv2.imread(import_from)
    #     undist = cv2.undistort(img, mtx, dist, None, mtx)
    #     export_to = 'camera_cal_undistorted/' + img_name
    #     #save the image in the destination folder#
    #     plt.imsave(export_to, undist)
            
    return mtx, dist         

### STEP 2: Perspective Transform from Car Camera to Bird's Eye View ###
# img_width = 1280
# img_heigt = 720

def warp(img, mtx, dist): # mts, dist
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    img_size = (img.shape[1], img.shape[0])
    #print(img_size)
    offset = 150
    
    # Source points taken from images with straight lane lines, 
    # these are to become parallel after the warp transform
    # src = np.float32([
    #     (350, 1080), # bottom-left corner
    #     (845, 700), # top-left corner 
    #     (1020, 700), # top-right corner
    #     (1560, 1080) # bottom-right corner
    # ])
    src = np.float32([
        (317, 720), # bottom-left corner
        (559, 457), # top-left corner 
        (671, 457), # top-right corner
        (1026, 720) # bottom-right corner
    ])
    # src = np.float32([
    #     (int(img_size[0]*350/1920), int(img_size[1]*1080/1080)), # bottom-left corner
    #     (int(img_size[0]*845/1920), int(img_size[1]*700/1080)), # top-left corner 
    #     (int(img_size[0]*1020/1920), int(img_size[1]*700/1080)), # top-right corner
    #     (int(img_size[0]*1560/1920), int(img_size[1]*1080/1080)) # bottom-right corner
    # ])
    # Destination points are to be parallel, taken into account the image size
    dst = np.float32([
        [offset, img_size[1]],             # bottom-left corner
        [offset, 0],                       # top-left corner
        [img_size[0]-offset, 0],           # top-right corner
        [img_size[0]-offset, img_size[1]]  # bottom-right corner
    ])
    # Calculate the transformation matrix and it's inverse transformation
    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(undist, M, img_size)
    
    return warped, M_inv, undist

def binary_thresholded(img):
    # Transform image to gray scale
    gray_img =cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply sobel (derivative) in x direction, this is usefull to detect lines that tend to be vertical
    sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    # Scale result to 0-255
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    sx_binary = np.zeros_like(scaled_sobel)
    # Keep only derivative values that are in the margin of interest
    sx_binary[(scaled_sobel >= 30) & (scaled_sobel <= 255)] = 1

    # Detect pixels that are white in the grayscale image
    white_binary = np.zeros_like(gray_img)
    white_binary[(gray_img > 200) & (gray_img <= 255)] = 1 #200,255

    # Convert image to HLS
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    H = hls[:,:,0]
    S = hls[:,:,2]
    sat_binary = np.zeros_like(S)
    # Detect pixels that have a high saturation value
    sat_binary[(S > 200) & (S <= 255)] = 1 #90 , 255

    hue_binary =  np.zeros_like(H)
    # Detect pixels that are yellow using the hue component
    hue_binary[(H > 15) & (H <= 25)] = 1 #10, 25

    # Combine all pixels detected above
    binary_1 = cv2.bitwise_or(sx_binary, white_binary)
    binary_2 = cv2.bitwise_or(hue_binary, sat_binary)
    binary = cv2.bitwise_or(binary_1, binary_2)
    #plt.imshow(binary, cmap='gray')

    return binary


### STEP 4: Detection of Lane Lines Using Histogram ###

def find_lane_pixels_using_histogram(binary_warped):
    
    # Center Area noise deleted : 1920*1080 #
    # x1=  int(1280*(500/1920))
    # x2=  int(1280*(1420/1920))
    # binary_warped[:, x1:x2]=0
    
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
 
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = int(1280*(100/1920))
    # Set minimum number of pixels found to recenter window
    minpix =  int(1280*(50/1920))

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
        
        # ## if scan windows added  
        # cv2.rectangle(window_img,(win_xleft_high,win_y_high),(win_xleft_low,win_y_low),(255,255,255),3)
        # cv2.rectangle(window_img,(win_xright_high,win_y_high),(win_xright_low,win_y_low),(255,255,255),3)
        # plt.imshow(window_img)
        # plt.show
        
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

    return leftx, lefty, rightx, righty


def fit_poly(binary_warped,leftx, lefty, rightx, righty):
    ### Fit a second order polynomial to each with np.polyfit() ###
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)   

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    
    return left_fit, right_fit, left_fitx, right_fitx, ploty


def draw_poly_lines(binary_warped, left_fitx, right_fitx, ploty):     
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
        
    margin = int(1280*(100/1920))
    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                              ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                              ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Center Line added ###
    center_line_pts = (left_line_pts + right_line_pts)/2
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (100, 100, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (100, 100, 0))   
    cv2.fillPoly(window_img, np.int_([center_line_pts]), (200, 100, 0))
    
    result = cv2.addWeighted(out_img, 1, window_img, 0.9, 0) #(0.3)
    
    # Plot the polynomial lines onto the image
    # plt.plot(left_fitx, ploty, color='green')
    # plt.plot(right_fitx, ploty, color='blue')
    ## End visualization steps ##

    return result


### STEP 5: Detection of Lane Lines Based on Previous Step ###

def find_lane_pixels_using_prev_poly(binary_warped):
    
    # global prev_left_fit
    # global prev_right_fit

    # width of the margin around the previous polynomial to search
    margin = int(1280*(100/1920))
    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])    
    ### Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    left_lane_inds = ((nonzerox > (prev_left_fit[0]*(nonzeroy**2) + prev_left_fit[1]*nonzeroy + 
                    prev_left_fit[2] - margin)) & (nonzerox < (prev_left_fit[0]*(nonzeroy**2) + 
                    prev_left_fit[1]*nonzeroy + prev_left_fit[2] + margin))).nonzero()[0]
    right_lane_inds = ((nonzerox > (prev_right_fit[0]*(nonzeroy**2) + prev_right_fit[1]*nonzeroy + 
                    prev_right_fit[2] - margin)) & (nonzerox < (prev_right_fit[0]*(nonzeroy**2) + 
                    prev_right_fit[1]*nonzeroy + prev_right_fit[2] + margin))).nonzero()[0]
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty


### STEP 6: Calculate Vehicle Position and Curve Radius ###

def measure_curvature_meters(binary_warped, left_fitx, right_fitx, ploty):
    # Define conversions in x and y from pixels space to meters
    # ym_per_pix = 30/1080 # meters per pixel in y dimension
    # xm_per_pix = 3.7/1920 # meters per pixel in x dimension
    ym_per_pix = 30/1080 *(720/1080)  # meters per pixel in y dimension
    xm_per_pix = 3.7/1920 *(1280/1920) # meters per pixel in x dimension
    
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    
    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    return left_curverad, right_curverad

def measure_position_meters(binary_warped, left_fit, right_fit):
    # Define conversion in x from pixels space to meters
    xm_per_pix = 3.7/1920 * (1280/1920)# meters per pixel in x dimension
    # Choose the y value corresponding to the bottom of the image
    y_max = binary_warped.shape[0]
    # Calculate left and right line positions at the bottom of the image
    left_x_pos = left_fit[0]*y_max**2 + left_fit[1]*y_max + left_fit[2]
    right_x_pos = right_fit[0]*y_max**2 + right_fit[1]*y_max + right_fit[2] 
    # Calculate the x position of the center of the lane 
    center_lanes_x_pos = (left_x_pos + right_x_pos)//2
    # Calculate the deviation between the center of the lane and the center of the picture
    # The car is assumed to be placed in the center of the picture
    # If the deviation is negative, the car is on the felt hand side of the center of the lane
    veh_pos = ((binary_warped.shape[1]//2) - center_lanes_x_pos) * xm_per_pix 
    return veh_pos


### STEP 7: Project Lane Delimitations Back on Image Plane and Add Text for Lane Info ###

def project_lane_info(img, binary_warped, ploty, left_fitx, right_fitx, M_inv, left_curverad, right_curverad, veh_pos):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    # Center Line modified
    margin = 400 * (1280/1920)
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    
    pts_left_c = np.array([np.transpose(np.vstack([left_fitx+margin, ploty]))])
    pts_right_c = np.array([np.flipud(np.transpose(np.vstack([right_fitx-margin, ploty])))])
    pts = np.hstack((pts_left_c, pts_right_c))
    
    pts_left_i = np.array([np.transpose(np.vstack([left_fitx+margin+150, ploty]))])
    pts_right_i = np.array([np.flipud(np.transpose(np.vstack([right_fitx-margin-150, ploty])))])
    pts_i = np.hstack((pts_left_i, pts_right_i))
    
    # Draw the lane onto the warped blank image
    colorwarp_img=cv2.polylines(color_warp, np.int_([pts_left]), False, (0,0, 255),50)
    colorwarp_img=cv2.polylines(color_warp, np.int_([pts_right]), False, (0,0, 255),50)
    colorwarp_img=cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    # colorwarp_img=cv2.fillPoly(color_warp, np.int_([pts_i]), (0,0, 255))
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, M_inv, (img.shape[1], img.shape[0]))
       
    # Combine the result with the original image
    out_img = cv2.addWeighted(img, 0.7, newwarp, 0.3, 0)
         
    cv2.putText(out_img,'Curve Radius [m]: '+str((left_curverad+right_curverad)/2)[:7],(5,80), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (255,255,255),2,cv2.LINE_AA)
    cv2.putText(out_img,'Center Offset [m]: '+str(veh_pos)[:7],(5,110), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,(255,255,255),2,cv2.LINE_AA)
    
    return out_img, colorwarp_img, newwarp


### STEP 8: Lane Finding Pipeline on Video ###

def lane_finding_pipeline(img,init, mts, dist):

    global left_fit_hist 
    global right_fit_hist
    global prev_left_fit
    global prev_right_fit
    
    if init:
        left_fit_hist = np.array([])
        right_fit_hist = np.array([])
        prev_left_fit = np.array([])
        prev_right_fit = np.array([])
        
    binary_thresh = binary_thresholded(img)
    binary_warped, M_inv, _ = warp(binary_thresh, mts, dist)

    ## checking ###
    binary_thresh_s = np.dstack((binary_thresh, binary_thresh, binary_thresh))*255
    binary_warped_s = np.dstack((binary_warped, binary_warped, binary_warped))*255
    
    if (len(left_fit_hist) == 0):
        leftx, lefty, rightx, righty = find_lane_pixels_using_histogram(binary_warped)
        left_fit, right_fit, left_fitx, right_fitx, ploty = fit_poly(binary_warped,leftx, lefty, rightx, righty)
        # Store fit in history
        left_fit_hist = np.array(left_fit)
        new_left_fit = np.array(left_fit)
        left_fit_hist = np.vstack([left_fit_hist, new_left_fit])
        right_fit_hist = np.array(right_fit)
        new_right_fit = np.array(right_fit)
        right_fit_hist = np.vstack([right_fit_hist, new_right_fit])
    
    else:
        prev_left_fit = [np.mean(left_fit_hist[:,0]), np.mean(left_fit_hist[:,1]), np.mean(left_fit_hist[:,2])]
        prev_right_fit = [np.mean(right_fit_hist[:,0]), np.mean(right_fit_hist[:,1]), np.mean(right_fit_hist[:,2])]
        leftx, lefty, rightx, righty = find_lane_pixels_using_prev_poly(binary_warped)
        if (len(lefty) == 0 or len(righty) == 0):
            leftx, lefty, rightx, righty = find_lane_pixels_using_histogram(binary_warped)
        left_fit, right_fit, left_fitx, right_fitx, ploty = fit_poly(binary_warped,leftx, lefty, rightx, righty)
        
        # Add new values to history
        new_left_fit = np.array(left_fit)
        left_fit_hist = np.vstack([left_fit_hist, new_left_fit])
        new_right_fit = np.array(right_fit)
        right_fit_hist = np.vstack([right_fit_hist, new_right_fit])
        
        # Remove old values from history
        if (len(left_fit_hist) > 5): #10
            left_fit_hist = np.delete(left_fit_hist, 0,0)
            right_fit_hist = np.delete(right_fit_hist, 0,0)
            
    ### chekcing ###
    draw_poly_img = draw_poly_lines(binary_warped, left_fitx, right_fitx, ploty)
                                       
    left_curverad, right_curverad =  measure_curvature_meters(binary_warped, left_fitx, right_fitx, ploty)
                                     #measure_curvature_meters(binary_warped, left_fitx, right_fitx, ploty)
    veh_pos = measure_position_meters(binary_warped, left_fit, right_fit) 
    out_img, colorwarp_img, newwarp = project_lane_info(img, binary_warped, ploty, left_fitx, right_fitx, M_inv, left_curverad, right_curverad, veh_pos)
    return out_img, veh_pos, colorwarp_img, draw_poly_img


def main():

    # cap = cv2.VideoCapture('/home/amrlabs/Documents/github/1v_Advanced-Lane-Detection/sample_driving_0621-2.mp4')
    cap = cv2.VideoCapture('./yolov5/data/video/test_sample.mp4') # test_sample.mp4
    # cap = cv2.VideoCapture('sample_driving_0621-2.mp4')
    if not cap.isOpened():
        print('File open failed!')
        cap.release()
        sys.exit()

    ## video out ##
    w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) 
    delay=int(1000 / fps)

    angle=0
    img_steering = cv2.imread('steering_wheel_image.jpg')
    rows,cols,ext= img_steering.shape

    # create the `VideoWriter()` object
    out = cv2.VideoWriter('./yolov5/data/output/result_output_lane.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    init=True
    mtx, dist = distortion_factors()

    while True:
        ret, frame =cap.read()

        if not ret:
            break
       
        img_out, angle, colorwarp, draw_poly_img = lane_finding_pipeline(frame, init, mtx, dist)

        if angle>1.5 or angle <-1.5:
            init=True
        else:
            init=False

        #Steering Image
        #angle = atan((180/pi)*(angle/5))
        M = cv2.getRotationMatrix2D((cols/2,rows/2),-angle*10,1)
        dst = cv2.warpAffine(img_steering,M,(cols,rows))
        #cv2.imshow("steering wheel", dst)
        height, width, channel = dst.shape
        height1, width1, channel1 = img_out.shape
        #img_out[(height1-height):height1, int(width1/2-width/2):(int(width1/2-width/2)+width)] = dst

        #Videowirte
        out.write(img_out)    
        
        cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
        cv2.imshow('frame', img_out)
        cv2.namedWindow('colorwarp',cv2.WINDOW_NORMAL)
        cv2.imshow('colorwarp', colorwarp)
        cv2.namedWindow('draw_poly',cv2.WINDOW_NORMAL)
        cv2.imshow('draw_poly', draw_poly_img)
    
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

