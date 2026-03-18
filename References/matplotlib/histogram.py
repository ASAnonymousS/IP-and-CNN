from matplotlib import pyplot

img_location = input('Enter the location of the image:\n')

img = pyplot.imread(img_location)

window, canvas = pyplot.subplots()
canvas.hist(img)
pyplot.show()
