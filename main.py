import cv2
import numpy as np
from problem import pixel_wise_matching, window_based_matching, window_based_matching_cosine

pixel_wise_matching('tsukuba/left.png', 'tsukuba/right.png', disparity_range=16,loss_type='l2_loss', file_name='tsukuba/prediction')
# window_based_matching(left_img_path='aloe/Aloe/Aloe_left_1.png', right_img_path='aloe/Aloe/Aloe_right_1.png', disparity_range=64,kernel_size=5, loss_type='l1_loss', file_name='aloe/Aloe/prediction_window')
# window_based_matching_cosine(left_img_path='aloe/Aloe/Aloe_left_1.png', 
#                              right_img_path='aloe/Aloe/Aloe_right_2.png', 
#                              disparity_range=64,
#                              kernel_size=5, file_name='aloe/Aloe/prediction_window')