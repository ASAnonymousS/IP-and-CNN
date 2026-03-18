from matplotlib import pyplot
import copy

img_location = input('Enter the location of the image:\n')

img1 = pyplot.imread(img_location)

window, canvas = pyplot.subplots(2, 2)


def lightness(img1):
    img = copy.deepcopy(img1)
    height, width, channel = img1.shape
    for i in range(width):
        for j in range(height):
            red = img1[i][j][0]
            green = img1[i][j][1]
            blue = img1[i][j][2]
            maximum = max(red, green, blue)
            minimum = min(red, green, blue)
            average = (maximum + minimum)/2
            if channel == 3:
                img[i][j] = [average, average, average]
            else:
                img[i][j] = [average, average, average, img[i][j][3]]
    return img


def average(img1):
    img = copy.deepcopy(img1)
    height, width, channel = img1.shape
    for i in range(width):
        for j in range(height):
            red = img1[i][j][0]
            green = img1[i][j][1]
            blue = img1[i][j][2]
            average = (red + green + blue)/3
            if channel == 3:
                img[i][j] = [average, average, average]
            else:
                img[i][j] = [average, average, average, img[i][j][3]]
    return img


def luminousity(img1):
    img = copy.deepcopy(img1)
    height, width, channel = img1.shape
    for i in range(width):
        for j in range(height):
            red = img1[i][j][0]
            green = img1[i][j][1]
            blue = img1[i][j][2]
            average = 0.299*red + 0.587*green + 0.114*blue
            if channel == 3:
                img[i][j] = [average, average, average]
            else:
                img[i][j] = [average, average, average, img[i][j][3]]
    return img


canvas[0, 0].imshow(img1)
canvas[0, 0].set_title('Original image')
canvas[0, 1].imshow(lightness(img1))
canvas[0, 1].set_title('Lightness method')
canvas[1, 0].imshow(average(img1))
canvas[1, 0].set_title('Average method')
canvas[1, 1].imshow(luminousity(img1))
canvas[1, 1].set_title('Luminousity method')
pyplot.show()
