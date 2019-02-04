
import math
import numpy as np
import tensorflow as tf


def just_distance(pt1, pt2):
    distance = math.sqrt((pt2[1]-pt1[1])**2+(pt2[0]-pt1[0])**2)
    if distance == 0:
        return -1
    return math.sqrt(distance)


def just_ndim_distance(pt1, pt2):
    distance = 0
    for j in range(len(pt1)):
        distance += (pt2[j]-pt1[j])**2
    if distance == 0:
        return -1
    return math.sqrt(distance)


def print_array(arr):
    print("the array ")
    for x in arr:
        print(x, ' ')


def k_weighted_nn(test_point, samples, class_quant, k, accuracy):
    # Weighted by the inverse of Euclidean distance
    dist_array = []
    sum_array = np.zeros(class_quant)
    for pt in samples:
        dist_array.append([ just_distance(test_point, pt), pt[2] ])
    dist_array = sorted(dist_array)
    if int(dist_array[0][0]) == -1:
        class_value = dist_array[0][1]
        # print('IGUAL')
    else:
        # print_array(dist_array)
        for c2 in range(k):
            sum_array[int(dist_array[c2][1])] += 1/dist_array[c2][0]
        # print_array(sum_array)
        max_val = sum_array[0]
        class_value = 0
        for ww in range(len(sum_array)):
            if sum_array[ww] > max_val:
                max_val = sum_array[ww]
                class_value = ww
    if int(test_point[2]) == int(class_value):
        accuracy += 1
    return accuracy
    # return 'The test point is class '+str(class_value)


def k_weighted_nn_cycle(samples, class_quant, k):
    acc = 0
    for point in samples:
        # Code to remove the test point from the samples

        samples_copy = samples
        for w in range(len(samples_copy)):
            if (point == samples_copy[w]).all():
                samples_copy = np.delete(samples_copy, w, 0)
                break
        # End of code to remove pt
        acc = k_weighted_nn(point, samples_copy, class_quant, k, acc)
    print('The accuracy of weighted knn algorithm is: '+str((acc*100)/len(samples)))

# Test point
# tp = [10, 10]
# CLASSES_QUANTITY = 2
# k_neighbors = 7
# L_C = 1
# # [x,y,class]
# data = [[1, 2, 0], [3, 4, 0], [5, 6, 1], [7, 7, 1], [2, 2, 0], [0, 2, 0], [7, 5, 1]]
# print(knn(tp, data, CLASSES_QUANTITY, k_neighbors))
