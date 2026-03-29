# read pickle file
import pickle
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

iterations = 5

# load pickle file
data = pickle.load(open('zsz-0040.png.p', 'rb'))  # 'rb' is read binary mode
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
TP = cv.bitwise_and(prediction, gt)
gtSubPrediction = cv.subtract(prediction, cv.bitwise_and(prediction, gt))
plt.imshow(gtSubPrediction, cmap='gray')
plt.title('False Positive')


plt.subplot(3, 3, 4)
# dilation operation
kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
dilated = cv.dilate(gt, kernel, iterations=1)
plt.imshow(dilated, cmap='gray')
plt.title('Dilated GT')
dilatedSubOriginal = cv.subtract(dilated, gt)

plt.subplot(3, 3, 5)
plt.imshow(dilatedSubOriginal, cmap='gray')
plt.title('Dilated - Original GT')

plt.subplot(3, 3, 6)
PredMultSubDilated = cv.bitwise_and(prediction, dilatedSubOriginal)
plt.imshow(PredMultSubDilated, cmap='gray')
plt.title('WLC-1st')

gt_2 = cv.bitwise_or(gt, PredMultSubDilated)
dilated = cv.dilate(gt_2, kernel, iterations=1)
dilatedSubOriginal = cv.subtract(dilated, gt)
PredMultSubDilated_2 = cv.bitwise_and(prediction, dilatedSubOriginal)
plt.subplot(3, 3, 7)
plt.imshow(PredMultSubDilated_2, cmap='gray')
plt.title('WLC-2nd')

plt.subplot(3, 3, 8)
_2Sub1 = cv.subtract(PredMultSubDilated_2, PredMultSubDilated)
plt.imshow(_2Sub1, cmap='gray')
plt.title('2 - 1')

visualization = np.ones((*gt.shape,3), dtype=np.uint8)  # * for unpacking the tuple
visualization = visualization * 255
diffMultPred_previous = None
for i in range(iterations):
    if diffMultPred_previous is not None:
        gt = cv.bitwise_or(gt, diffMultPred_previous)
    dilated = cv.dilate(gt, kernel, iterations=1)
    dilatedSubGT = cv.subtract(dilated, gt)
    diffMultPred = cv.bitwise_and(prediction, dilatedSubGT)

    # if i == 0:
    #     plt.imshow(diffMultPred)
    #     plt.title('WLC-1st')
    #     plt.show()
    # else:
    #     mask = cv.subtract(diffMultPred, diffMultPred_previous)
    visualization[diffMultPred > 0] = (np.array(color_array[i][:3]) * 255).astype(np.uint8)
    diffMultPred_previous = diffMultPred

plt.subplot(3, 3, 9)
plt.imshow(visualization, cmap='hot')
plt.title(f'WLC-Visualization {iterations} iterations')


plt.tight_layout()

plt.show(interpolation='nearest')
plt.savefig('output_image.png', format='png', dpi=300, bbox_inches='tight', pad_inches=0)
plt.close()
