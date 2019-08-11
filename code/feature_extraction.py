import cv2
import numpy as np


def feature_extraction(img, feature):
    """
    This function computes defined feature (HoG, SIFT) descriptors of the target image.

    :param img: a height x width x channels matrix,
    :param feature: name of image feature representation.

    :return: a N x feature_size matrix.
    """

    if feature == 'HoG':
        # HoG parameters

        # In the case of the Hog Feature, we already given the base parameters for using hog feature function.
        # TA - You can just use that parameter with each subdivide image (which has image grid size * image grid size)
        # Thank you for the reply. Does it mean to divide the image into 20x20 size sub-images and perform the feature extraction on each image??
        # TA - Yes. In the SIFT, image grid size is different.

        win_size = (32, 32)
        block_size = (32, 32)
        block_stride = (16, 16)
        cell_size = (16, 16)

        nbins = 9
        deriv_aperture = 1
        win_sigma = 4
        histogram_norm_type = 0
        l2_hys_threshold = 2.0000000000000001e-01
        gamma_correction = 0
        nlevels = 64

        # Your code here. You should also change the return value.

        # sample visualizing
        # cv2.imshow('img', img)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


        hog = cv2.HOGDescriptor(win_size,
                                block_size,
                                block_stride,
                                cell_size,
                                nbins,
                                deriv_aperture,
                                win_sigma,
                                histogram_norm_type,
                                l2_hys_threshold,
                                gamma_correction,
                                nlevels)

        # additional parameters

        #hist = hog.compute(gray,winStride,padding,locations)

        #TODO: Check if this is valid???

        hist = hog.compute(gray)
        hist_resized = np.resize(hist, (int(len(hist)/36), 36))
        hist_resized
        return hist_resized

    elif feature == 'SIFT':

        # Your code here. You should also change the return value.

        #input image size 240 * 200 ==> divide H, W by 20 ==> 12 * 10 = 120
        #in case of this input image, the number of feature is 120.
        #So the number of feature is changed according to input image size.

        #IF PROBLEMS WITH DEPENDENCIES: pip3 install opencv-contrib-python==3.4.2.16

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(gray, None)

        return des




