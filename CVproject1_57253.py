import cv2
import numpy as np

# Reading the salt and pepper image
filename = 'N3.png'
img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
# showing the image's pixel height and width
print(img.shape)  # Prints the shape of the image
height, width = img.shape # putting the pixel's length in height and width accordingly
cv2.namedWindow('original')
cv2.imshow('original', img)  # Shows the before image
cv2.waitKey(0)

# Making a function to apply median blur filter on original image
def median_filter(img, size):
    # run a loop from half of the size + 1 to  up to
    # number of rows present in the image
    for i in range(size // 2 + 1, img.shape[0]):

        # run a loop  from half of the size + 1 up to
        # number of columns present in the image
        for j in range(size // 2 + 1, img.shape[1]):
            # Take a sub-matrix of specified size form img image matrix
            N = img[i - size // 2: i + size // 2 + 1, j - size // 2: j + size // 2 + 1]

            # find out median of sub_matrix
            med = np.median(N)

            # assign that medium value to the specified pixel coordinates
            img[i - 2, j - 2] = med

    # return blur image
    return img


#median_filter function calling
img1 = median_filter(img, 5)
# displaying the smoothed image
cv2.namedWindow('filtered')
cv2.imshow('filtered', img1)
cv2.imwrite('filteredN3.png',img1)
cv2.waitKey(0)

# Reading the filtered image
filename = 'filteredN3.png'
img1 = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

# Threshold for N3 and making it binary
ret, thresh = cv2.threshold(img1, 60, 255, cv2.THRESH_BINARY)
img_binary = thresh
cv2.namedWindow('BINARY')
cv2.imshow('BINARY', img_binary)  # Shows the image in binary
# cv2.imwrite('binaryN3.png', img_binary) # saves the binary image
cv2.waitKey(0)

# Applying the medianBlur filter for better results
img2 = cv2.medianBlur(img_binary, 5, 0)
cv2.imshow('main2', img2)
# cv2.imwrite('n3man.png', img2)
cv2.waitKey(0)

# Setting up Structuring element
strel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))

# Opening the binary image to get better results
img2 = cv2.morphologyEx(img2, cv2.MORPH_ERODE, strel)
cv2.namedWindow('calc2', )
cv2.imshow('calc2', img2) # showing the opened image
cv2.waitKey(0)

filenamef = 'NF3.png'
imgF = cv2.imread(filenamef, cv2.IMREAD_GRAYSCALE)
# Threshold for NF3 and making it binary
ret, thresh = cv2.threshold(imgF, 60, 255, cv2.THRESH_BINARY)
img_binaryf = thresh
cv2.namedWindow('BINARYF')
cv2.imshow('BINARYF', img_binaryf)
# cv2.imwrite('binaryNF3.png', img_binary)
cv2.waitKey(0)

# Making the summed area table function
def summed_area_table(img):
    # creating a table full of zeroes
    table = np.zeros_like(img).astype(int)
    # going through all the table of the image
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            # adding every point above and left of every table's note
            if (row > 0) and (col > 0):
                table[row, col] = (img[row, col] +
                                   table[row, col - 1] +
                                   table[row - 1, col] -
                                   table[row - 1, col - 1])
            elif row > 0:
                table[row, col] = img[row, col] + table[row - 1, col]
            elif col > 0:
                table[row, col] = img[row, col] + table[row, col - 1]
            else:
                table[row, col] = img[row, col]

    return table

# Calling the summed area table function
summed_table = summed_area_table(imgF)
# print (summed_table)
# Calling the find contours function and saving every table in contours
(_, contours, _) = cv2.findContours(image=img2,
                                    mode=cv2.RETR_EXTERNAL,
                                    method=cv2.CHAIN_APPROX_SIMPLE)
# Running a loop for every contour found
count = 0
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if x != 0 and y != 0 and (x + w) != width and (y + h) != height: # Making sure to process only whole objects
        cnt_len = cv2.arcLength(cnt, True) # perimeter lenght of each object
        cnt = cv2.approxPolyDP(cnt, 0.02 * cnt_len, True)
        area = cv2.contourArea(cnt) #area of each object
        # Calling the boundingRect function to take the 4 points surrounding every object in a rectangle shape
        x, y, w, h = cv2.boundingRect(cnt)
        #print(x + h, x + w, y + h, y + w)
        # Allocating the sum of points of every boundary window
        sum = summed_table[y + h, x + w] - summed_table[y, x + w] - summed_table[y + h, x] + summed_table[y, x]
        mean_sum= sum/(w*h)
        # Raise the count only if an object whole withing the image is found
        count = count + 1
        print('the area of the', count, 'object, is', int(area))
        print('the area of the', count,' boundary window is', w * h, 'and the mean of the bounding window is',
              mean_sum)

print('The total objects that are whole in the image are', count)
