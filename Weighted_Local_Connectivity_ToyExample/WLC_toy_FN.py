# read pickle file
import pickle
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

iterations = 9

# load pickle file
data = pickle.load(open('sch-0036.png.p', 'rb'))  # 'rb' is read binary mode
# assign data to variables
# print(data.keys())
# keys: dict_keys(['input', 'output', 'ground_truth', 'dice', 'iou'])

gt = data['ground_truth']
prediction = data['output']
gt = gt.astype(np.uint8)
prediction = prediction.astype(np.uint8)

plt.figure(figsize=(10, 10))
color_map = plt.get_cmap('hot')
color_array = [color_map(i / iterations) for i in range(iterations)]
print('color_array: ', color_array)


plt.subplot(3, 3, 1)
plt.imshow(gt, cmap='gray')
plt.title('Ground Truth')

plt.subplot(3, 3, 2)
plt.imshow(prediction, cmap='gray')
plt.title('Prediction')

plt.subplot(3, 3, 3)
gtSubPrediction = cv.subtract(gt, cv.bitwise_and(gt, prediction))
plt.imshow(gtSubPrediction, cmap='gray')
plt.title('False Negative')

plt.subplot(3, 3, 4)
# dilation operation
kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
dilated = cv.dilate(prediction, kernel, iterations=1)
plt.imshow(dilated, cmap='gray')
plt.title('Dilated Prediction')
dilatedSubOriginal = cv.subtract(dilated, prediction)

plt.subplot(3, 3, 5)
plt.imshow(dilatedSubOriginal, cmap='gray')
plt.title('Dilated - Original Prediction')
# print('gt shape: ', gt.shape)
# print('dilatedSubOriginal shape: ', dilatedSubOriginal.shape)

plt.subplot(3, 3, 6)
gtMultSubDilated = cv.bitwise_and(gt, dilatedSubOriginal)
plt.imshow(gtMultSubDilated, cmap='gray')
plt.title('WLC-1st')


dilated = cv.dilate(prediction, kernel, iterations=2)
dilatedSubOriginal = cv.subtract(dilated, prediction)
gtMultSubDilated_2 = cv.bitwise_and(gt, dilatedSubOriginal)
plt.subplot(3, 3, 7)
plt.imshow(gtMultSubDilated_2, cmap='gray')
plt.title('WLC-2nd')

plt.subplot(3, 3, 8)
_2Sub1 = cv.subtract(gtMultSubDilated_2, gtMultSubDilated)
plt.imshow(_2Sub1, cmap='gray')
plt.title('2 - 1')

visualization = np.ones((*gt.shape,3), dtype=np.uint8)  # * for unpacking the tuple
visualization = visualization * 255
diffMultGt_previous = None
for i in range(iterations):
    if diffMultGt_previous is not None:
        prediction = cv.bitwise_or(prediction, diffMultGt_previous)
    dilated = cv.dilate(prediction, np.ones((3, 3), np.uint8), iterations=1)
    dilatedSubPrediction = cv.subtract(dilated, prediction)
    diffMultGt = cv.bitwise_and(gt, dilatedSubPrediction)
    visualization[diffMultGt > 0] = (np.array(color_array[i][:3]) * 255).astype(np.uint8)
    diffMultGt_previous = diffMultGt

plt.subplot(3, 3, 9)
plt.imshow(visualization)
plt.title(f'WLC-Visualization {iterations} iterations')


plt.tight_layout()

plt.show()