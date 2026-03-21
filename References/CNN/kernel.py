import cv2
import numpy

# kernel = numpy.array([
#     [1, 1, 1],
#     [1, -8, 1],
#     [1, 1, 1]
# ])

# Random Weight Generation
kernelR = numpy.array(numpy.random.uniform(-1, 1, (3, 3)))
kernelG = numpy.array(numpy.random.uniform(-1, 1, (3, 3)))
kernelB = numpy.array(numpy.random.uniform(-1, 1, (3, 3)))
bias1 = numpy.random.uniform(-1, 1, 1)

img1 = cv2.imread("../Image_Samples/Lenna.png", cv2.IMREAD_COLOR)

height, width, channel = img1.shape
imgR = img1[:, :, 2]
imgG = img1[:, :, 1]
imgB = img1[:, :, 0]

img2 = numpy.zeros([height - 2, width - 2], dtype=numpy.uint8)
for h in range(1, height - 1):
    for i in range(1, width - 1):
        # Convolution begins
        for j in range(-1, 2):
            for k in range(-1, 2):
                img2[h - 1, i - 1] += (imgR[h + j, i + k] * kernelR[j + 1, k + 1]) + (imgG[h + j, i + k] * kernelG[j + 1, k + 1]) + (imgB[h + j, i + k] * kernelB[j + 1, k + 1])
        # Convolution ends
        # Bias Added
        img2[h - 1, i - 1] += bias1[0]
        # RELU Function Applied
        img2[h - 1, i - 1] = max(0, img2[h - 1, i - 1])

kernel2 = numpy.array(numpy.random.uniform(-1, 1, (3, 3)))
bias2 = numpy.random.uniform(-1, 1, 1)

height, width = img2.shape

img3 = numpy.zeros([height - 2, width - 2], dtype=numpy.uint8)
for h in range(1, height - 1):
    for i in range(1, width - 1):
        # Convolution begins
        for j in range(-1, 2):
            for k in range(-1, 2):
                img3[h - 1, i - 1] += (img2[h + j, i + k] * kernel2[j + 1, k + 1])
        # Convolution ends
        # Bias Added
        img3[h - 1, i - 1] += bias2[0]
        # RELU Function Applied
        img3[h - 1, i - 1] = max(0, img3[h - 1, i - 1])

# Max Pool: (Stride = 2)
height, width = img3.shape

img4 = numpy.zeros([(height + 1) // 2, (width + 1) // 2], dtype=numpy.uint8)
for i in range(0, height, 2):
    for j in range(0, width, 2):
        img4[i // 2, j // 2] = numpy.max(img3[i: (i + 3), j: (j + 3)])

# --- Layer 4: Convolution 3 + ReLU (Results in img5) ---
kernel3 = numpy.random.uniform(-1, 1, (3, 3))
bias3 = numpy.random.uniform(-1, 1, 1)

h4, w4 = img4.shape  # Input is the pooled image img4
img5 = numpy.zeros([h4 - 2, w4 - 2], dtype=numpy.uint8)

for h in range(1, h4 - 1):
    for i in range(1, w4 - 1):
        # Convolution begins
        for j in range(-1, 2):
            for k in range(-1, 2):
                img5[h - 1, i - 1] += (img4[h + j, i + k] * kernel3[j + 1, k + 1])
        # Bias and ReLU
        img5[h - 1, i - 1] = max(0, img5[h - 1, i - 1] + bias3[0])

# --- Layer 5: Convolution 4 + ReLU (Results in img6) ---
kernel4 = numpy.random.uniform(-1, 1, (3, 3))
bias4 = numpy.random.uniform(-1, 1, 1)

h5, w5 = img5.shape
img6 = numpy.zeros([h5 - 2, w5 - 2], dtype=numpy.uint8)

for h in range(1, h5 - 1):
    for i in range(1, w5 - 1):
        # Convolution begins
        for j in range(-1, 2):
            for k in range(-1, 2):
                img6[h - 1, i - 1] += (img5[h + j, i + k] * kernel4[j + 1, k + 1])
        # Bias and ReLU
        img6[h - 1, i - 1] = max(0, img6[h - 1, i - 1] + bias4[0])

# --- Layer 6: Final Max Pool (Results in img7) ---
# Stride = 2, Window = 2x2
h6, w6 = img6.shape
img7 = numpy.zeros([(h6 + 1) // 2, (w6 + 1) // 2], dtype=numpy.uint8)

for i in range(0, h6, 2):
    for j in range(0, w6, 2):
        # Taking max of 2x2 neighborhood
        img7[i // 2, j // 2] = numpy.max(img6[i: i + 2, j: j + 2])

print(img1.shape)
print(img2.shape)
print(img3.shape)
print(img4.shape)
print(img5.shape)
print(img6.shape)
print(img7.shape)
# with numpy.printoptions(threshold = numpy.inf):
#     print(img1)
#     print(img4)
cv2.imshow("Lenna Image", img1,)
cv2.imshow("Kernel 1", img2)
cv2.imshow("Kernel 2", img3)
cv2.imshow("Max Pooling 1", img4)
cv2.imshow("Kernel 3", img5)
cv2.imshow("Kernel 4", img6)
cv2.imshow("Max Pooling 2", img7)
cv2.waitKey(0)
cv2.destroyAllWindows()


##############################################################################################################################################################################################################################
########################################################################## THE CODE THAT YOU WILL BE READING NOW IS GENERATED VIA GEMINI TO SAVE TIME ########################################################################

# AI Gen 1:
# import cv2
# import numpy
#
# # Random Weight Generation
# kernelR = numpy.array(numpy.random.uniform(-1, 1, (3, 3)))
# kernelG = numpy.array(numpy.random.uniform(-1, 1, (3, 3)))
# kernelB = numpy.array(numpy.random.uniform(-1, 1, (3, 3)))
# bias1 = numpy.random.uniform(-1, 1, 1)
#
# img1 = cv2.imread("../Image_Samples/Lenna.png", cv2.IMREAD_COLOR)
#
# height, width, channel = img1.shape
# imgR = img1[:, :, 2]
# imgG = img1[:, :, 1]
# imgB = img1[:, :, 0]
#
# img2 = numpy.zeros([height - 2, width - 2], dtype=numpy.uint8)
# for h in range(1, height - 1):
#     for i in range(1, width - 1):
#         # Convolution begins
#         for j in range(-1, 2):
#             for k in range(-1, 2):
#                 # Using cv2.multiply and cv2.add for uint8 operations
#                 valR = cv2.multiply(imgR[h + j, i + k], kernelR[j + 1, k + 1])
#                 valG = cv2.multiply(imgG[h + j, i + k], kernelG[j + 1, k + 1])
#                 valB = cv2.multiply(imgB[h + j, i + k], kernelB[j + 1, k + 1])
#
#                 sum_rgb = cv2.add(cv2.add(valR, valG), valB)
#                 img2[h - 1, i - 1] = cv2.add(img2[h - 1, i - 1], sum_rgb)
#         # Convolution ends
#         # Bias Added
#         img2[h - 1, i - 1] = cv2.add(img2[h - 1, i - 1], bias1[0].astype(numpy.uint8))
#         # RELU Function Applied
#         img2[h - 1, i - 1] = max(0, img2[h - 1, i - 1])
#
# kernel2 = numpy.array(numpy.random.uniform(-1, 1, (3, 3)))
# bias2 = numpy.random.uniform(-1, 1, 1)
#
# height, width = img2.shape
#
# img3 = numpy.zeros([height - 2, width - 2], dtype=numpy.uint8)
# for h in range(1, height - 1):
#     for i in range(1, width - 1):
#         # Convolution begins
#         for j in range(-1, 2):
#             for k in range(-1, 2):
#                 val = cv2.multiply(img2[h + j, i + k], kernel2[j + 1, k + 1])
#                 img3[h - 1, i - 1] = cv2.add(img3[h - 1, i - 1], val)
#         # Convolution ends
#         # Bias Added
#         img3[h - 1, i - 1] = cv2.add(img3[h - 1, i - 1], bias2[0].astype(numpy.uint8))
#         # RELU Function Applied
#         img3[h - 1, i - 1] = max(0, img3[h - 1, i - 1])
#
# # Max Pool: (Stride = 2)
# height, width = img3.shape
#
# img4 = numpy.zeros([(height + 1) // 2, (width + 1) // 2], dtype=numpy.uint8)
# for i in range(0, height, 2):
#     for j in range(0, width, 2):
#         img4[i // 2, j // 2] = numpy.max(img3[i: (i + 3), j: (j + 3)])
#
# # --- Layer 4: Convolution 3 + ReLU (Results in img5) ---
# kernel3 = numpy.random.uniform(-1, 1, (3, 3))
# bias3 = numpy.random.uniform(-1, 1, 1)
#
# h4, w4 = img4.shape
# img5 = numpy.zeros([h4 - 2, w4 - 2], dtype=numpy.uint8)
#
# for h in range(1, h4 - 1):
#     for i in range(1, w4 - 1):
#         for j in range(-1, 2):
#             for k in range(-1, 2):
#                 val = cv2.multiply(img4[h + j, i + k], kernel3[j + 1, k + 1])
#                 img5[h - 1, i - 1] = cv2.add(img5[h - 1, i - 1], val)
#         img5[h - 1, i - 1] = cv2.add(img5[h - 1, i - 1], bias3[0].astype(numpy.uint8))
#         img5[h - 1, i - 1] = max(0, img5[h - 1, i - 1])
#
# # --- Layer 5: Convolution 4 + ReLU (Results in img6) ---
# kernel4 = numpy.random.uniform(-1, 1, (3, 3))
# bias4 = numpy.random.uniform(-1, 1, 1)
#
# h5, w5 = img5.shape
# img6 = numpy.zeros([h5 - 2, w5 - 2], dtype=numpy.uint8)
#
# for h in range(1, h5 - 1):
#     for i in range(1, w5 - 1):
#         for j in range(-1, 2):
#             for k in range(-1, 2):
#                 val = cv2.multiply(img5[h + j, i + k], kernel4[j + 1, k + 1])
#                 img6[h - 1, i - 1] = cv2.add(img6[h - 1, i - 1], val)
#         img6[h - 1, i - 1] = cv2.add(img6[h - 1, i - 1], bias4[0].astype(numpy.uint8))
#         img6[h - 1, i - 1] = max(0, img6[h - 1, i - 1])
#
# # --- Layer 6: Final Max Pool (Results in img7) ---
# h6, w6 = img6.shape
# img7 = numpy.zeros([(h6 + 1) // 2, (w6 + 1) // 2], dtype=numpy.uint8)
#
# for i in range(0, h6, 2):
#     for j in range(0, w6, 2):
#         img7[i // 2, j // 2] = numpy.max(img6[i: i + 2, j: j + 2])
#
# print(img1.shape)
# print(img2.shape)
# print(img3.shape)
# print(img4.shape)
# print(img5.shape)
# print(img6.shape)
# print(img7.shape)
#
# cv2.imshow("Lenna Image", img1)
# cv2.imshow("Kernel 1", img2)
# cv2.imshow("Kernel 2", img3)
# cv2.imshow("Max Pooling 1", img4)
# cv2.imshow("Kernel 3", img5)
# cv2.imshow("Kernel 4", img6)
# cv2.imshow("Max Pooling 2", img7)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


##############################################################################################################################################################################################################################


# AI Gen 2:
# import cv2
# import numpy
#
# # Random Weight Generation
# kernelR = numpy.array(numpy.random.uniform(-1, 1, (3, 3)))
# kernelG = numpy.array(numpy.random.uniform(-1, 1, (3, 3)))
# kernelB = numpy.array(numpy.random.uniform(-1, 1, (3, 3)))
# bias1 = numpy.random.uniform(-1, 1, 1)
#
# img1 = cv2.imread("../Image_Samples/Lenna.png", cv2.IMREAD_COLOR)
#
# height, width, channel = img1.shape
# imgR = img1[:, :, 2]
# imgG = img1[:, :, 1]
# imgB = img1[:, :, 0]
#
# # --- Layer 1: RGB Convolution ---
# img2 = numpy.zeros([height - 2, width - 2], dtype=numpy.uint8)
# for h in range(1, height - 1):
#     for i in range(1, width - 1):
#         pixel_sum = 0.0
#         for j in range(-1, 2):
#             for k in range(-1, 2):
#                 valR = imgR[h + j, i + k] * kernelR[j + 1, k + 1]
#                 valG = imgG[h + j, i + k] * kernelG[j + 1, k + 1]
#                 valB = imgB[h + j, i + k] * kernelB[j + 1, k + 1]
#                 pixel_sum += (valR + valG + valB)
#
#         pixel_sum += bias1[0]
#         # Clip handles both Saturation (0-255) and ReLU (max 0)
#         img2[h - 1, i - 1] = numpy.clip(pixel_sum, 0, 255).astype(numpy.uint8)
#
# # --- Layer 2: Convolution ---
# kernel2 = numpy.array(numpy.random.uniform(-1, 1, (3, 3)))
# bias2 = numpy.random.uniform(-1, 1, 1)
#
# h2, w2 = img2.shape
# img3 = numpy.zeros([h2 - 2, w2 - 2], dtype=numpy.uint8)
# for h in range(1, h2 - 1):
#     for i in range(1, w2 - 1):
#         pixel_sum = 0.0
#         for j in range(-1, 2):
#             for k in range(-1, 2):
#                 pixel_sum += img2[h + j, i + k] * kernel2[j + 1, k + 1]
#
#         pixel_sum += bias2[0]
#         img3[h - 1, i - 1] = numpy.clip(pixel_sum, 0, 255).astype(numpy.uint8)
#
# # --- Layer 3: Max Pool 1 (Stride = 2) ---
# h3, w3 = img3.shape
# img4 = numpy.zeros([(h3 + 1) // 2, (w3 + 1) // 2], dtype=numpy.uint8)
# for i in range(0, h3, 2):
#     for j in range(0, w3, 2):
#         img4[i // 2, j // 2] = numpy.max(img3[i: i + 2, j: j + 2])
#
# # --- Layer 4: Convolution 3 ---
# kernel3 = numpy.random.uniform(-1, 1, (3, 3))
# bias3 = numpy.random.uniform(-1, 1, 1)
#
# h4, w4 = img4.shape
# img5 = numpy.zeros([h4 - 2, w4 - 2], dtype=numpy.uint8)
# for h in range(1, h4 - 1):
#     for i in range(1, w4 - 1):
#         pixel_sum = 0.0
#         for j in range(-1, 2):
#             for k in range(-1, 2):
#                 pixel_sum += img4[h + j, i + k] * kernel3[j + 1, k + 1]
#
#         pixel_sum += bias3[0]
#         img5[h - 1, i - 1] = numpy.clip(pixel_sum, 0, 255).astype(numpy.uint8)
#
# # --- Layer 5: Convolution 4 ---
# kernel4 = numpy.random.uniform(-1, 1, (3, 3))
# bias4 = numpy.random.uniform(-1, 1, 1)
#
# h5, w5 = img5.shape
# img6 = numpy.zeros([h5 - 2, w5 - 2], dtype=numpy.uint8)
# for h in range(1, h5 - 1):
#     for i in range(1, w5 - 1):
#         pixel_sum = 0.0
#         for j in range(-1, 2):
#             for k in range(-1, 2):
#                 pixel_sum += img5[h + j, i + k] * kernel4[j + 1, k + 1]
#
#         pixel_sum += bias4[0]
#         img6[h - 1, i - 1] = numpy.clip(pixel_sum, 0, 255).astype(numpy.uint8)
#
# # --- Layer 6: Final Max Pool (Stride = 2) ---
# h6, w6 = img6.shape
# img7 = numpy.zeros([(h6 + 1) // 2, (w6 + 1) // 2], dtype=numpy.uint8)
# for i in range(0, h6, 2):
#     for j in range(0, w6, 2):
#         img7[i // 2, j // 2] = numpy.max(img6[i: i + 2, j: j + 2])
#
# # Results
# print(f"Original: {img1.shape}")
# print(f"Conv1:    {img2.shape}")
# print(f"Conv2:    {img3.shape}")
# print(f"Pool1:    {img4.shape}")
# print(f"Conv3:    {img5.shape}")
# print(f"Conv4:    {img6.shape}")
# print(f"Pool2:    {img7.shape}")
#
# cv2.imshow("Lenna Image", img1)
# cv2.imshow("Kernel 1", img2)
# cv2.imshow("Kernel 2", img3)
# cv2.imshow("Max Pooling 1", img4)
# cv2.imshow("Kernel 3", img5)
# cv2.imshow("Kernel 4", img6)
# cv2.imshow("Max Pooling 2", img7)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
