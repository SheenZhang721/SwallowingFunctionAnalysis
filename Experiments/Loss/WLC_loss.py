# code for weighted local connectivity loss
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class WLC_loss(nn.Module):
    def __init__(self, iterations = 5, visualize = False):
        super(WLC_loss, self).__init__()
        self.iterations = iterations
        self.distance_weight = torch.tensor([i ** 2 for i in range(1, self.iterations+1)], dtype=torch.float32)
        self.visualize = visualize

    def forward(self, prediction, ground_truth):
        self.distance_weight = self.distance_weight.to(prediction.device)


        FN_total = 0.0
        false_negative = None

        for i in range(self.iterations):
            if false_negative is not None:
                prediction = prediction + false_negative
            # dilate the prediction to get the false negative
            dilated_prediction = torch.nn.functional.conv2d(
                prediction,
                weight = torch.ones(1, 1, 3, 3).to(prediction.device),
                padding = 1
            ).squeeze().clamp(0, 1)  # clamp to make sure the value is between 0 and 1

            sharp_perdition = torch.sigmoid(20 * (dilated_prediction - prediction - 0.5))
            false_negative = ground_truth * sharp_perdition
            FN_sum = torch.sum(false_negative)
            FN_total += self.distance_weight[i] * FN_sum/torch.sum(ground_truth)

            if self.visualize:
                # visualize the false negative
                plt.figure(figsize=(10, 10))
                plt.subplot(3, 3, 1)
                plt.imshow(ground_truth.cpu().numpy().squeeze(), cmap='gray')
                plt.title('Ground Truth')
                plt.subplot(3, 3, 2)
                plt.imshow(prediction.cpu().numpy().squeeze(), cmap='gray')
                plt.title('Prediction')
                plt.subplot(3, 3, 3)
                plt.imshow(dilated_prediction.cpu().numpy().squeeze(), cmap='gray')
                plt.title('Dilated Prediction')
                plt.subplot(3, 3, 4)
                plt.imshow((dilated_prediction-prediction).cpu().numpy().squeeze(), cmap='gray')
                plt.title('dilated_prediction-prediction')
                plt.subplot(3, 3, 5)
                plt.imshow(sharp_perdition.cpu().numpy().squeeze(), cmap='gray')
                plt.title('Sharp Prediction')
                plt.subplot(3, 3, 6)
                plt.imshow(false_negative.cpu().numpy().squeeze(), cmap='gray')
                plt.title('False Negative')
                plt.show()
                print(f'iteration: {i+1}', 'FN_total: ', FN_total, 'FN_sum: ', FN_sum)

        return FN_total

# test the WLC_loss
if __name__ == '__main__':
    import pickle
    import os
    # read image as tensor
    # print(os.getcwd())
    p_file = pickle.load(open('../../Weighted_Local_Connectivity_ToyExample/sch-0036.png.p', 'rb'))
    gt = torch.tensor(p_file['ground_truth']).unsqueeze(0).unsqueeze(0).float()
    prediction = torch.tensor(p_file['output']).unsqueeze(0).unsqueeze(0).float()
    # create the loss object
    loss = WLC_loss(iterations=12, visualize=True)
    # calculate the loss
    loss_value = loss(prediction, gt)
    print('loss_value: ', loss_value)