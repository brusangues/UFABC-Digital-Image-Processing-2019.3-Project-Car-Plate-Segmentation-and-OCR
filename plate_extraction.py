import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import pickle

#from TOOLS import Functions
class ifChar:
    # this function contains some operations used by various function in the code
    def __init__(self, cntr):
        self.contour = cntr

        self.boundingRect = cv2.boundingRect(self.contour)

        [x, y, w, h] = self.boundingRect

        self.boundingRectX = x
        self.boundingRectY = y
        self.boundingRectWidth = w
        self.boundingRectHeight = h

        self.boundingRectArea = self.boundingRectWidth * self.boundingRectHeight

        self.centerX = (self.boundingRectX + self.boundingRectX + self.boundingRectWidth) / 2
        self.centerY = (self.boundingRectY + self.boundingRectY + self.boundingRectHeight) / 2

        self.diagonalSize = math.sqrt((self.boundingRectWidth ** 2) + (self.boundingRectHeight ** 2))

        self.aspectRatio = float(self.boundingRectWidth) / float(self.boundingRectHeight)

class PossiblePlate:
    def __init__(self):
        self.Plate = None
        self.Grayscale = None
        self.Thresh = None

        self.rrLocationOfPlateInScene = None

        self.strChars = ""


# this function is a 'first pass' that does a rough check on a contour to see if it could be a char
def checkIfChar(possibleChar):
    if (possibleChar.boundingRectArea > 40 and possibleChar.boundingRectWidth > 2
            and possibleChar.boundingRectHeight > 8 and 0.10 < possibleChar.aspectRatio < 1.5):
    #80,2,8,.25,1.0
        return True
    else:
        return False


# check the center distance between characters
def distanceBetweenChars(firstChar, secondChar):
    x = abs(firstChar.centerX - secondChar.centerX)
    y = abs(firstChar.centerY - secondChar.centerY)

    return math.sqrt((x ** 2) + (y ** 2))


# use basic trigonometry (SOH CAH TOA) to calculate angle between chars
def angleBetweenChars(firstChar, secondChar):
    adjacent = float(abs(firstChar.centerX - secondChar.centerX))
    opposite = float(abs(firstChar.centerY - secondChar.centerY))

    # check to make sure we do not divide by zero if the center X positions are equal
    # float division by zero will cause a crash in Python
    if adjacent != 0.0:
        angleInRad = math.atan(opposite / adjacent)
    else:
        angleInRad = 1.5708

    # calculate angle in degrees
    angleInDeg = angleInRad * (180.0 / math.pi)

    return angleInDeg

# Classificação de Caracteres

def knn_load(file_path):
    with np.load(file_path) as data:
        #print( data.files )
        train = data['train']
        train_labels = data['train_labels']
        knn_classes = data['classes']
    
    knn = cv2.ml.KNearest_create()
    knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)
    return knn, knn_classes

def knn_classify(img, model, knn_classes):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    try:
        img_resized = cv2.resize(img ,(20,20), interpolation = cv2.INTER_NEAREST)
    except:
        return "#"
    #if np.mean(img) < 127:
    #    img = 255-img
        #ret, img = cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    img_reshaped = img_resized.reshape(-1,400).astype(np.float32)
    ret,result,neighbours,dist = model.findNearest(img_reshaped,k=5)
    
    #plt.figure(figsize=(4,4))
    #plt.subplot(121); plt.title("img"); plt.imshow(img,"gray")
    #plt.subplot(122); plt.title(knn_classes[int(result)]); plt.imshow(img_resized,"gray")
    #plt.show()
    return knn_classes[int(result)]

# Função auxiliar pequena

def img_glue(img1, img2):
	img2 = img2.astype(np.float64) / np.max(img2) # normalize the data to 0 - 1
	img2 = 255 * img2 # Now scale by 255
	img2 = img2.astype(np.uint8)
	y1,x1 = img1.shape[:2]
	y2,x2 = img2.shape[:2]
	img3 = np.zeros([max(y1,y2), x1+x2, 3], dtype=np.uint8)
	img3[:y1, :x1, :] = img1
	img3[:y2, x1:x1+x2, :] = img2
	return img3

#Função de extração de placas principal

def extract_plate(img):
    img_original = img.copy()
    ### CONTOURS________________________________________________________________###
    # hsv transform - value = gray image
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)

    # kernel to use for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    square3 = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    round3 = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    round5 = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
    round5[1:4, 1:4] = square3

    # applying topHat/blackHat operations
    #topHat = cv2.morphologyEx(value.copy(),cv2.MORPH_OPEN, round3, iterations = 3)
    #topHat = cv2.subtract(value.copy(), topHat)
    #blackHat = cv2.morphologyEx(value.copy(),cv2.MORPH_CLOSE, round3, iterations = 3)
    #blackHat= cv2.subtract(blackHat,value.copy())
    topHat = cv2.morphologyEx(value, cv2.MORPH_TOPHAT, kernel)
    blackHat = cv2.morphologyEx(value, cv2.MORPH_BLACKHAT, kernel)

    # add and subtract between morphological operations
    add = cv2.add(value, topHat)
    subtract = cv2.subtract(add, blackHat)
    
    # applying gaussian blur on subtract image
    blur = cv2.GaussianBlur(subtract, (5, 5), 0)

    # thresholding
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 9)

    # cv2.findCountours()
    cv2MajorVersion = cv2.__version__.split(".")[0]
    if int(cv2MajorVersion) >= 4:
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    else:
        imageContours, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # create a numpy array with shape given by threshed image value dimensions
    height, width = thresh.shape
    imageContours = np.zeros((height, width, 3), dtype=np.uint8)

    ### CHARS________________________________________________________________###
    # list and counter of possible chars
    possibleChars = []
    countOfPossibleChars = 0
    # loop to check if any (possible) char is found
    #print("contourslen", len(contours), cv2.arcLength(contours[0], True))
    for i in range(0, len(contours)):
        if cv2.arcLength(contours[i], True)<8: continue
        # draw contours based on actual found contours of thresh image
        cv2.drawContours(imageContours, contours, i, (255, 255, 255))

        # retrieve a possible char by the result ifChar class give us
        possibleChar = ifChar(contours[i])

        # by computing some values (area, width, height, aspect ratio) possibleChars list is being populated
        if checkIfChar(possibleChar) is True:
            countOfPossibleChars = countOfPossibleChars + 1
            possibleChars.append(possibleChar)

    ### PLATE________________________________________________________________###
    plates_list = []
    listOfListsOfMatchingChars = []
    for possibleC in possibleChars:
        
        # the purpose of this function is, given a possible char and a big list of possible chars,
        # find all chars in the big list that are a match for the single possible char, and return those 
        # matching chars as a list
        def matchingChars(possibleC, possibleChars):
            listOfMatchingChars = []
            
            # if the char we attempting to find matches for is the exact same char as the char in the big list we are currently checking
            # then we should not include it in the list of matches b/c that would end up double including the current char
            # so do not add to list of matches and jump back to top of for loop
            for possibleMatchingChar in possibleChars:
                if possibleMatchingChar == possibleC:
                    continue

                # compute stuff to see if chars are a match
                distChars= distanceBetweenChars(possibleC, possibleMatchingChar)

                angleChars= angleBetweenChars(possibleC, possibleMatchingChar)

                changeInArea = float(abs(possibleMatchingChar.boundingRectArea - possibleC.boundingRectArea)) / float(
                    possibleC.boundingRectArea)

                changeInWidth = float(abs(possibleMatchingChar.boundingRectWidth - possibleC.boundingRectWidth)) / float(
                    possibleC.boundingRectWidth)

                changeInHeight = float(abs(possibleMatchingChar.boundingRectHeight - possibleC.boundingRectHeight)) / float(
                    possibleC.boundingRectHeight)

                # check if chars match
                if distChars< (possibleC.diagonalSize * 5) and \
                        angleChars< 12.0 and \
                        changeInArea < 0.5 and \
                        changeInWidth < 0.8 and \
                        changeInHeight < 0.2:
                    listOfMatchingChars.append(possibleMatchingChar)

            return listOfMatchingChars

        # here we are re-arranging the one big list of chars into a list of lists of matching chars
        # the chars that are not found to be in a group of matches do not need to be considered further
        listOfMatchingChars = matchingChars(possibleC, possibleChars)

        listOfMatchingChars.append(possibleC)

        # if current possible list of matching chars is not long enough to constitute a possible plate
        # jump back to the top of the for loop and try again with next char
        if len(listOfMatchingChars) < 3:
            continue

        # here the current list passed test as a "group" or "cluster" of matching chars
        listOfListsOfMatchingChars.append(listOfMatchingChars)

        # remove the current list of matching chars from the big list so we don't use those same chars twice,
        # make sure to make a new big list for this since we don't want to change the original big list
        listOfPossibleCharsWithCurrentMatchesRemoved = list(set(possibleChars) - set(listOfMatchingChars))

        recursiveListOfListsOfMatchingChars = []

        for recursiveListOfMatchingChars in recursiveListOfListsOfMatchingChars:
            listOfListsOfMatchingChars.append(recursiveListOfMatchingChars)

        break

    ### ROTATION________________________________________________________________###
    for listOfMatchingChars in listOfListsOfMatchingChars:
        possiblePlate = PossiblePlate()

        # sort chars from left to right based on x position
        listOfMatchingChars.sort(key=lambda matchingChar: matchingChar.centerX)

        # calculate the center point of the plate
        plateCenterX = (listOfMatchingChars[0].centerX + listOfMatchingChars[len(listOfMatchingChars) - 1].centerX) / 2.0
        plateCenterY = (listOfMatchingChars[0].centerY + listOfMatchingChars[len(listOfMatchingChars) - 1].centerY) / 2.0
        plateCenter = plateCenterX, plateCenterY

        # calculate plate width and height
        plateWidth = int((listOfMatchingChars[len(listOfMatchingChars) - 1].boundingRectX + listOfMatchingChars[
            len(listOfMatchingChars) - 1].boundingRectWidth - listOfMatchingChars[0].boundingRectX) * 1.3)

        totalOfCharHeights = 0

        for matchingChar in listOfMatchingChars:
            totalOfCharHeights = totalOfCharHeights + matchingChar.boundingRectHeight

        averageCharHeight = totalOfCharHeights / len(listOfMatchingChars)

        plateHeight = int(averageCharHeight * 1.5)

        # calculate correction angle of plate region
        opposite = listOfMatchingChars[len(listOfMatchingChars) - 1].centerY - listOfMatchingChars[0].centerY

        hypotenuse = distanceBetweenChars(listOfMatchingChars[0],
                                                    listOfMatchingChars[len(listOfMatchingChars) - 1])
        correctionAngleInRad = math.asin(opposite / hypotenuse)
        correctionAngleInDeg = correctionAngleInRad * (180.0 / math.pi)

        # pack plate region center point, width and height, and correction angle into rotated rect member variable of plate
        possiblePlate.rrLocationOfPlateInScene = (tuple(plateCenter), (plateWidth, plateHeight), correctionAngleInDeg)

        # get the rotation matrix for our calculated correction angle
        rotationMatrix = cv2.getRotationMatrix2D(tuple(plateCenter), correctionAngleInDeg, 1.0)

        height, width, numChannels = img.shape

        # rotate the entire image
        imgRotated = cv2.warpAffine(img, rotationMatrix, (width, height))

        # crop the image/plate detected
        imgCropped = cv2.getRectSubPix(imgRotated, (plateWidth-20, plateHeight), tuple(plateCenter))

        # copy the cropped plate image into the applicable member variable of the possible plate
        possiblePlate.Plate = imgCropped

        # populate plates_list with the detected plate
        if possiblePlate.Plate is not None:
            plates_list.append(possiblePlate)

        # draw a ROI on the original image
        for i in range(0, len(plates_list)):
            # finds the four vertices of a rotated rect - it is useful to draw the rectangle.
            p2fRectPoints = cv2.boxPoints(plates_list[i].rrLocationOfPlateInScene)

            # roi rectangle colour
            rectColour = (0, 255, 0)

            cv2.line(imageContours, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), rectColour, 2)
            cv2.line(imageContours, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), rectColour, 2)
            cv2.line(imageContours, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), rectColour, 2)
            cv2.line(imageContours, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), rectColour, 2)

            cv2.line(img, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), rectColour, 2)
            cv2.line(img, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), rectColour, 2)
            cv2.line(img, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), rectColour, 2)
            cv2.line(img, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), rectColour, 2)

            ###cv2.imshow("detected", imageContours)
            # cv2.imwrite(temp_folder + '11 - detected.png', imageContours)

            ###cv2.imshow("detectedOriginal", img)
            # cv2.imwrite(temp_folder + '12 - detectedOriginal.png', img)

            # cv2.imshow("plate", plates_list[i].Plate)
            # cv2.imwrite(temp_folder + '13 - plate.png', plates_list[i].Plate)
    
    if len(listOfListsOfMatchingChars)>0:
        return img, imgCropped
    return img, np.zeros([10,10])
    ###cv2.waitKey(0)

#Classificação de caracteres nas placas

def extract_plate_chars(img_plate):
    #Convertendo para escala de cinza
    img_gray = cv2.cvtColor(img_plate,cv2.COLOR_BGR2GRAY)
    
    #Aplicando Threshold OTSU
    thresh, img_thresh = cv2.threshold(img_gray,0,255,cv2.THRESH_OTSU+cv2.THRESH_BINARY_INV)
    #plt.figure(figsize=(7,7)); plt.title("img_thresh"); fig = plt.imshow(img_thresh,"gray")

    #Extraindo contornos
    contours = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    
    #Selecionando contornos que são Caracteres
    chars = [contour for contour in contours if 40<cv2.arcLength(contour, True)<150 
            and abs(cv2.contourArea(contour, True))<370 
            and cv2.boundingRect(contour)[0]>2
            and cv2.boundingRect(contour)[0]<150 ]
    #print("contours", len(contours))
    #print("chars", len(chars))
    #img_out = img.copy()
    #img_out = cv2.drawContours(img_out, chars, -1, (255,0,0), 1)
    #plt.figure(figsize=(7,7)); plt.title("plate"); fig = plt.imshow(img_out,"gray")

    #Classificando os Caracteres
    m_letters, c_letters = knn_load('data\knn_data_letters.npz')
    m_numbers, c_numbers = knn_load('data\knn_data_numbers.npz')

    #Ordenação dos Caracteres
    chars = sorted(chars, key=lambda char: cv2.boundingRect(char)[0] )

    plate_list = []
    #Iterando sobre os contornos e classificando
    for i in range(len(chars)):
        (x,y,w,h) = cv2.boundingRect(chars[i])
        x-=2; y-=2; w+=4; h+=4
        
        #Segmentando
        img_shape = np.ones((x+w,y+h))
        img_shape = img_plate[y:y+h, x:x+w, :]
        #img_shape = cv2.morphologyEx(img_shape,cv2.MORPH_OPEN,round3, iterations = 1)
        #img_shape = cv2.dilate(img_shape,square3,iterations=1)

        if i<3:
            result = knn_classify(img_shape, m_letters, c_letters)
        else:
            result = knn_classify(img_shape, m_numbers, c_numbers)
        
        plate_list.append(result)

        #print("result", result)

        #img_out = cv2.rectangle(img_out, (x,y), (x+w,y+h), [0,0,255], 2)
        
        cv2.putText(img_plate, result , (x, y+10), cv2.FONT_HERSHEY_PLAIN, 1, [255,0,255], 1)

    plate = ''.join(plate_list)

    return plate

#import matplotlib.pyplot as plt
#img_original = cv2.imread("data\plate5.jpg")

#img_plate = img_original.copy()
#img_plate, imgCropped = extract_plate(img_plate)
#plt.figure(figsize=(7,7)); plt.title("img_plate"); fig = plt.imshow(imgCropped,"gray")