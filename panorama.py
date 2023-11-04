import cv2
import streamlit as st
st.set_option('deprecation.showfileUploaderEncoding', False)
import numpy as np
import imutils
from PIL import Image

class Stitcher:
    def __init__(self):
        self.isv3 = imutils.is_cv3(or_better=True)

    def stitch(self, images, ratio=0.75, reprojThresh=4.0, showMatches=False):
        (imageB, imageA) = images
        (kpsA, featuresA) = self.detectAndDescribe(imageA)
        (kpsB, featuresB) = self.detectAndDescribe(imageB)
        M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)
        if M is None:
            return None

        (matches, H, status) = M
        result = cv2.warpPerspective(imageA, H,
                                     (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
        result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
        if showMatches:
            vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches, status)
            return (result, vis)
        return result

    def detectAndDescribe(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.isv3:
          sift = cv2.SIFT_create()
          (kps, features) = sift.detectAndCompute(image, None)
        else:
          detector = cv2.FeatureDetector_create("SIFT")
          kps = detector.detect(gray)
          extractor = cv2.DescriptorExtractor_create("SIFT")
          (kps, features) = extractor.compute(gray, kps)
        # arrays
        kps = np.float32([kp.pt for kp in kps])
        return (kps, features)

    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB,
                   ratio, reprojThresh):
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []

        for m in rawMatches:
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        if len(matches) > 4:
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
                                            reprojThresh)
            return (matches, H, status)

        return None

    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB
        for ((trainIdx, queryIdx), s) in zip(matches, status):
          # matched
          if s == 1:
            # draw the match
            ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
            ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
            cv2.line(vis, ptA, ptB, (0, 255, 0), 1)
        # return the visualization
        return vis



st.title("Image Stitching App Using Opencv")

# Sidebar
st.sidebar.title("Upload Images")

# Upload left and right images separately
left_image = st.sidebar.file_uploader("Upload left image", type=["jpg", "jpeg", "png"], accept_multiple_files=False )
if left_image is not None:
    left_image_pil = Image.open(left_image)
    left_image_np = np.array(left_image_pil)
    left_image_resized = imutils.resize(left_image_np, width=400)
    st.sidebar.subheader("Original Left Image")
    st.sidebar.image(left_image_pil, use_column_width=True)
right_image = st.sidebar.file_uploader("Upload right image", type=["jpg", "jpeg", "png"], accept_multiple_files=False)

if left_image is not None and right_image is not None:
    # Read and resize the uploaded images
    right_image_pil = Image.open(right_image)
    right_image_np = np.array(right_image_pil)
    right_image_resized = imutils.resize(right_image_np, width=400)
    
    st.sidebar.subheader("Original Right Image")
    st.sidebar.image(right_image_pil, use_column_width=True)
    # Stitch the left and right images together to create a panorama
    stitcher = Stitcher()
    (result, vis) = stitcher.stitch([left_image_resized, right_image_resized], showMatches=True)

    # Display the original left and right images and the stitched panorama

    if result is not None:
        st.subheader("Stitched Panorama Output:")
        st.image(result, use_column_width=True)
    else:
        st.error("Panorama stitching failed. Please ensure there are enough matching keypoints.")
else:
    st.warning("Please upload both left and right images for stitching.")

