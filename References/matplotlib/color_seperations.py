from matplotlib import pyplot

img_location = input('Enter the location of the image:\n')

img = pyplot.imread(img_location)


def RGB(img):
    height, width, channel = img.shape
    imgR = img.copy()
    imgG = img.copy()
    imgB = img.copy()

    # Red color:
    for i in range(height):
        for j in range(width):
            if channel == 3:
                imgR[i][j] = (imgR[i][j][0], 0, 0)
                imgG[i][j] = (0, imgG[i][j][1], 0)
                imgB[i][j] = (0, 0, imgB[i][j][2])
            elif channel == 4:
                imgR[i][j] = (imgR[i][j][0], 0, 0, imgR[i][j][3])
                imgG[i][j] = (0, imgG[i][j][1], 0, imgG[i][j][3])
                imgB[i][j] = (0, 0, imgB[i][j][2], imgB[i][j][3])

    window, canvas = pyplot.subplots(2, 2)
    canvas[0, 0].imshow(img)
    canvas[0, 1].imshow(imgR)
    canvas[1, 0].imshow(imgG)
    canvas[1, 1].imshow(imgB)
    pyplot.show()


RGB(img)
