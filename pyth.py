import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the two images
img1 = cv2.imread("graf/img1.ppm")   # source
img5 = cv2.imread("graf/img4.ppm")   # destination

# Convert to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray5 = cv2.cvtColor(img5, cv2.COLOR_BGR2GRAY)

# 1. Detect SIFT keypoints and descriptors
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp5, des5 = sift.detectAndCompute(gray5, None)

# 2. Match descriptors with BFMatcher + Lowe’s ratio test
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des5, k=2)

good_matches = []
src_pts = []
dst_pts = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)
        src_pts.append(kp1[m.queryIdx].pt)
        dst_pts.append(kp5[m.trainIdx].pt)

src_pts = np.float32(src_pts).reshape(-1, 1, 2)
dst_pts = np.float32(dst_pts).reshape(-1, 1, 2)

# 3. Estimate homography with RANSAC
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
print("Estimated Homography:\n", H)

# 4. Warp img1 into img5’s coordinate system
h5, w5 = img5.shape[:2]
warped = cv2.warpPerspective(img1, H, (w5, h5))

# 5. Simple overlay (replace nonzero pixels of warped onto img5)
stitched = img5.copy()
mask_warped = (warped > 0)
stitched[mask_warped] = warped[mask_warped]

# 6. Show results
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
plt.title("Warped img1 → img5 frame")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(cv2.cvtColor(stitched, cv2.COLOR_BGR2RGB))
plt.title("Stitched Image")
plt.axis("off")
plt.show()
