import cv2
import numpy as np
import imutils
import pytesseract
import re

def empty_cell(sudoku):
    '''
    INPUT: sudoku board
    OUTPUT: first empty cell found
    Return None if every cell has a number
    '''
    for i in range(len(sudoku)):
        for j in range(len(sudoku[0])):
            if sudoku[i][j] == 0:
                return (i, j)
    return None

def solvable(sudoku, number, position):
    '''
    INPUT: sudoku board
    OUTPUT: True if the sudoku can be solved, False otherwise
    '''
    # Check row
    for i in range(len(sudoku[0])):
        if sudoku[position[0]][i] == number and position[1] != i:
            return False

    # Check column
    for i in range(len(sudoku)):
        if sudoku[i][position[1]] == number and position[0] != i:
            return False

    # Check box
    box_x = position[1] // 3
    box_y = position[0] // 3

    for i in range(box_y*3, box_y*3 + 3):
        for j in range(box_x * 3, box_x*3 + 3):
            if sudoku[i][j] == number and (i,j) != position:
                return False

    return True

def solve_sudoku(sudoku):
    find = empty_cell(sudoku)
    if not find:
        return True
    else:
        row, col = find

    for i in range(1,10):
        if solvable(sudoku, i, (row, col)):
            sudoku[row][col] = i

            if solve_sudoku(sudoku):
                return True
            sudoku[row][col] = 0
    return False

def get_solved_sudoku(sudoku):
    '''
    INPUT: unsolved sudoku board
    OUTPUT: solved board if solvable, ValueError if not
    '''
    if solve_sudoku(sudoku):
        return sudoku
    else:
        raise ValueError

def perspective_img(img, loc):
    '''
    INPUT: image and location of interesting region
    OUTPUT: selected region with a perspective transformation
    '''
    h = 900
    w = 900
    p1 = np.float32([loc[0], loc[3], loc[1], loc[2]])
    p2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])

    # Apply Perspective Transform Algorithm
    matrix = cv2.getPerspectiveTransform(p1, p2)
    result = cv2.warpPerspective(img, matrix, (w, h))
    return result

# Read image
img=cv2.imread('sudoku1.png')
cv2.imshow('Orginal Image', img)
cv2.waitKey(-1)

''' 
    sudoku1.png --> 1 wrong detection
    sudoku2.png --> 0 wrong detections
    sudoku3.png --> multiple wrong detections     
'''

# Convert to grayscale
gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Grayscale Image', gray)
cv2.imwrite('Grayscale.png',gray)
cv2.waitKey(-1)

# Filter the image
bfilter = cv2.bilateralFilter(gray, 13, 20, 20)
cv2.imshow('Bilateral Filter', bfilter)
cv2.imwrite('BilateralFilter.png',bfilter)
cv2.waitKey(-1)

# Edge detection
canny = cv2.Canny(bfilter, 30, 200)
cv2.imshow( "Canny Edge Detection (30,200)", canny )
cv2.waitKey(-1)

# Dilation
k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
imageDil = cv2.dilate(canny, k, iterations=1)
cv2.imshow( "Image Dilation", imageDil )
cv2.waitKey(-1)

keypoints = cv2.findContours(imageDil.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

contours = imutils.grab_contours(keypoints)
newimg = cv2.drawContours(img.copy(), contours, -1, (0, 0, 255), 3)
cv2.imshow("Contour", newimg)
cv2.waitKey(-1)

contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]
location = None
# Finds rectangular contour
for contour in contours:
    approx = cv2.approxPolyDP(contour, 15, True)
    if len(approx) == 4:
        location = approx
        break
result = perspective_img(img, location)


cv2.imshow("Press 'r' to rotate by 90 Degrees", result)
k=cv2.waitKey(0)
rotation=0
while k != ord('q'):
    if k==ord('r'):
        result = imutils.rotate(result, 90)
        cv2.imshow("Press 'r' to rotate by 90 Degrees", result)
        k=cv2.waitKey(0)
        rotation+=1


# split the board into 81 individual images
def find_boxes(board):
    '''
    INPUT: sudoku board
    OUTPUT: 81 elements representing every cell
    '''
    rows = np.vsplit(board,9) # vertical split
    elements = []
    for r in rows:
        cols = np.hsplit(r,9) # horizontal split for every row
        for cell in cols:
            size_cell=cell.shape[0]
            cell = cv2.resize(cell,(size_cell, size_cell))/255.0
            elements.append(cell)
    return elements

gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
rois = find_boxes(gray)

# recognize image
l=[]
for box in rois:
    res=[]
    img_float32 = np.float32(box)
    box1 = cv2.cvtColor(img_float32, cv2.COLOR_BGRA2RGB)
    box1=box1*255 #senza queso passaggio e il successivo l'immagine si salva solo nera per un problema di normalizzazione
    box1=box1.astype(np.uint8)

    # Cropping the image
    box2 = box1[10:box1.shape[0]-10, 10:box1.shape[1]-10]

    # Thresholding
    _,thresh1 = cv2.threshold(box2,100,255,cv2.THRESH_BINARY)

    num=pytesseract.image_to_string(thresh1,config ='--psm 10')
    temp = re.findall(r'\d+', num)
    res = list(map(int, temp))

    cv2.imshow('Box',thresh1)

    # check if they are correct
    if len(res)==0:
        res.append(0)
    if res[0]>9:
        res[0]=res[0]-10
    numDet=np.ones(shape=(100,100,3))
    cv2.putText(numDet,text=str(res[0]),org=(30,80),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=2,color=(0,0,0),thickness=3)
    cv2.putText(numDet,text='Detected:',org=(0,15),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.7,color=(0,0,0),thickness=1)
    cv2.imshow('Detected number',numDet)
    
    k=cv2.waitKey(0)
    if k==ord('q'):
        l.append(res[0])
    if k==ord('0'):
        l.append(0)
    if k==ord('1'):
        l.append(1)
    if k==ord('2'):
        l.append(2)
    if k==ord('3'):
        l.append(3)
    if k==ord('4'):
        l.append(4)
    if k==ord('5'):
        l.append(5)
    if k==ord('6'):
        l.append(6)
    if k==ord('7'):
        l.append(7)
    if k==ord('8'):
        l.append(8)
    if k==ord('9'):
        l.append(9)
    
predicted_numbers=l

input_size = 9
rois = np.array(rois).reshape(-1, input_size, input_size, 1)

def displayNumbers(img, numbers, color=(0, 0, 255)):
    """Displays 81 numbers in an image or mask at the same position of each cell of the board"""
    W = int(img.shape[1]/9)
    H = int(img.shape[0]/9)
    for i in range (9):
        for j in range (9):
            if numbers[(j*9)+i] !=0:
                cv2.putText(img, str(numbers[(j*9)+i]), (i*W+int(W/2)-int((W/4)), int((j+0.7)*H)), cv2.FONT_HERSHEY_COMPLEX, 2, color, 2, cv2.LINE_AA)
    return img

def get_InvPerspective(img, masked_num, location, height = 900, width = 900):
    """Takes original image as input"""
    pts1 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    pts2 = np.float32([location[0], location[3], location[1], location[2]])

    # Apply Perspective Transform Algorithm
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(masked_num, matrix, (img.shape[1], img.shape[0]))
    return result

# reshape the list
board_num = np.array(predicted_numbers).astype('uint8').reshape(9, 9)

# solve the sudoku
try:    
    solved_sudoku = get_solved_sudoku(board_num)
    # array of the predicted numbers
    # 0: unsolved numbers of sudoku 
    # 1: given number
    array = np.where(np.array(predicted_numbers)>0, 0, 1)
    # get only solved numbers for the solved sudoku
    flat_solved_sudoku = solved_sudoku.flatten()*array
    # mask
    mask = np.zeros_like(result)
    # displays solved numbers in the mask in the same position where board numbers are empty
    sudoku_mask = displayNumbers(mask, flat_solved_sudoku)
    cv2.imshow("Solved Mask", sudoku_mask)
    cv2.waitKey(0)
    for i in range(rotation):
        sudoku_mask = imutils.rotate(sudoku_mask, -90)
    inv_mask = get_InvPerspective(img, sudoku_mask, location)
    cv2.imshow("Inverse Perspective Mask", inv_mask)
    cv2.waitKey(0)
    combined = cv2.addWeighted(img, 0.5, inv_mask, 1, 0)
    cv2.imshow("Final result", combined)
    cv2.waitKey(-1)

except:
    print('Not solved')