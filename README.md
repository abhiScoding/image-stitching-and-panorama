# Objectives
Task 1: Stitch two images<br />
Task 2: Stitch multiple images to generate a panorama view<br />


## Task1 Image Stitching
- Extracted SIFT features from both the images
- Stored key points and descriptors in dictionaries to correlate key points and associated descriptor
- Performed matching using sum of squared difference (SSD) of descriptors. Considering number of good matches for different SSD threshold and its effect on homography matrix, threshold is calculated.
- Performed homography by considering good matching points.
- Padded image 1 before warping it so that it won’t get cropped after warping/transform. The diameter of image 1 selected as amount of padding to generalize the code and to have maximum space available for warping.
- The warped image 1 and image 2 are overlapped considering distance between centroid of best matched feature points. Centroid of these feature points are calculated in both images and then image 2 is translated by distance between centroids in the warped image1.

## Task2: Image Panorama 
- Overlap array is calculated by considering number of best matching points between images. If image pairs have good matching points grater than the threshold than the pair considered as part of panorama. The threshold is tuned later based on quality of panorama.
- Using image matching relationship in the overlap array, matching image pairs are stitched by following the same method as task 1 to create a panorama.
- SSD threshold and min number of matches are calculated by trial and error (by analyzing effect on overlap array, and quality of panorama) on different set of images. These values might need tuning for other set of images for best result.
