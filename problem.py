import cv2
import numpy as np

def distance_l1(x, y):
    return np.abs(x - y)

def distance_l2(x, y):
    return np.sqrt((x - y)**2)

loss_dict = {
    "l1_loss":{
        "distance":distance_l1,
        "cost_max":255,
    },
    "l2_loss":{
        "distance":distance_l2,
        "cost_max":255,
    }
}

def pixel_wise_matching(left_img_path, right_img_path, disparity_range, loss_type="l1_loss", file_name=False):
    left_image_path = left_img_path
    left_image = cv2.imread(left_image_path)
    left_grey_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
    left_grey_image = left_grey_image.astype(np.float32)


    right_image_path = right_img_path
    right_image = cv2.imread(right_image_path)
    right_grey_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
    right_grey_image = right_grey_image.astype(np.float32)

    h , w =  left_grey_image.shape
    disparity_map = np.zeros((left_grey_image.shape), dtype=np.float32)

    
    for i in range(h):
        for j in range(w):
            
            if j < disparity_range:
                cost = np.full((disparity_range,), loss_dict[loss_type]['cost_max'])
                cost [:j+1] = loss_dict[loss_type]['distance'](left_grey_image[i,j] , right_grey_image[i, j::-1])
                disparity_map[i,j] = np.argmin(cost) * (255/disparity_range)
                
            else:
                disparity_map[i,j] = np.argmin(loss_dict[loss_type]['distance'](left_grey_image[i,j] ,right_grey_image[i, j : j-disparity_range:-1])) * (255/disparity_range)
            
    if file_name:
        print('Saving result...')
        # Save results
        cv2.imwrite(f'{file_name}_{loss_type}.png', disparity_map.astype(np.uint8))
        cv2.imwrite(f'{file_name}_{loss_type}_color.png', cv2.applyColorMap(disparity_map.astype(np.uint8), cv2.COLORMAP_JET))
        print('Done.')

    return disparity_map.astype(np.uint8)

def window_based_matching(left_img_path, right_img_path, disparity_range, kernel_size=3, loss_type="l1_loss", file_name=False):
    left_image_path = left_img_path
    left_image = cv2.imread(left_image_path)
    left_grey_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
    left_grey_image = left_grey_image.astype(np.float32)


    right_image_path = right_img_path
    right_image = cv2.imread(right_image_path)
    right_grey_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
    right_grey_image = right_grey_image.astype(np.float32)

    h , w =  left_grey_image.shape
    radius = (kernel_size-1)//2
    disparity_map = np.zeros((left_grey_image.shape), dtype=np.float32)

    
    # for i in range(radius, h-radius):
    #     for j in range(radius+disparity_range, w-radius):
    for i in range(radius,h-radius):
        for j in range(radius,w- radius):
            #i, j = 2 ,2 
           # cost = []
            max_cost = np.inf
            d_optimal = 0
            for d in range(disparity_range):
                # total = 0
                # for v in range(-radius, radius+1):
                #     for u in range(-radius, radius+1):
                #         total += 255 if (j+u-d < 0) else distance_l1(left_grey_image[i+v, j+u], right_grey_image[i+v, j+u-d])
                # if total < max_cost:
                #     max_cost = total
                #     d_optimal1 = d
                
                left_kernel = left_grey_image[i-radius:i+radius+1, j-radius:j+radius+1]
                right_kernel = np.full((kernel_size, kernel_size),255)
                if j-d - radius >= 0:
                    right_kernel = right_grey_image[i-radius:i+radius+1, j - d - radius :j - d + radius+1]
                    #print('test', right_kernel.shape, j - d - radius , j - d + radius+1)
                elif j-d - radius < 0  and j -d -radius > - kernel_size:
                    #print(right_grey_image[i-radius:i+radius+1, np.abs(j -d -1 ):kernel_size], j, d)
                    right_kernel[:,np.abs(j -d -1 ):] = right_grey_image[i-radius:i+radius+1, np.abs(j -d -1 ):kernel_size]
                    # print(right_grey_image[i-radius:i+radius+1, np.abs(j -d -1 ):kernel_size])
                #print(j, d, right_kernel.shape, radius)
                total = np.sum(np.abs(left_kernel-right_kernel))
                if total < max_cost:
                    max_cost = total
                    d_optimal = d
                # if d_optimal1!=d_optimal:
                #     print(d_optimal1, d_optimal)
            disparity_map[i,j] = d_optimal * (255/disparity_range)
            
        
                
                 
    if file_name:
        print('Saving result...')
        # Save results
        cv2.imwrite(f'{file_name}_{loss_type}.png', disparity_map.astype(np.uint8))
        cv2.imwrite(f'{file_name}_{loss_type}_color.png', cv2.applyColorMap(disparity_map.astype(np.uint8), cv2.COLORMAP_JET))
        print('Done.')

    return disparity_map.astype(np.uint8)

def window_based_matching_cosine(left_img_path, right_img_path, disparity_range, kernel_size=3, file_name=False):
    left_image_path = left_img_path
    left_image = cv2.imread(left_image_path)
    left_grey_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
    left_grey_image = left_grey_image.astype(np.float32)


    right_image_path = right_img_path
    right_image = cv2.imread(right_image_path)
    right_grey_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
    right_grey_image = right_grey_image.astype(np.float32)

    h , w =  left_grey_image.shape
    radius = (kernel_size-1)//2
    disparity_map = np.zeros((left_grey_image.shape), dtype=np.float32)

    

    for i in range(radius, h-radius):
        for j in range(radius, w- radius):

            max_cost = -1
            d_optimal = 0 
            for d in range(disparity_range):
                total = 0
                left_kernel = left_grey_image[i-radius:i+radius+1, j-radius:j+radius+1]
                right_kernel = np.full((kernel_size, kernel_size),255)
                if j-d - radius >= 0:
                    right_kernel = right_grey_image[i-radius:i+radius+1, j - d - radius :j - d + radius+1]
                elif j-d - radius < 0  and j -d -radius > - kernel_size:
                    right_kernel[:,np.abs(j -d -1 ):] = right_grey_image[i-radius:i+radius+1, np.abs(j -d -1 ):kernel_size]

                left_vector = left_kernel.flatten()
                right_vector = right_kernel.flatten()
                total = np.dot(left_vector, right_vector)/(np.linalg.norm(left_vector)*np.linalg.norm(right_vector))
                if total > max_cost:
                    max_cost = total
                    d_optimal = d
                
            disparity_map[i,j] = d_optimal * (255/disparity_range)
            
                 
    if file_name:
        print('Saving result...')
        # Save results
        cv2.imwrite(f'{file_name}_cosine_similarity.png', disparity_map.astype(np.uint8))
        cv2.imwrite(f'{file_name}__cosine_similarity_color.png', cv2.applyColorMap(disparity_map.astype(np.uint8), cv2.COLORMAP_JET))
        print('Done.')

    return disparity_map.astype(np.uint8)
