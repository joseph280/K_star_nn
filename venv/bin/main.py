
import math
import numpy as np
import tensorflow as tf


def just_distance(x1,y1,x2,y2):
    distance = math.sqrt((y2-y1)**2+(x2-x1)**2)
    return L_C*distance


def print_array(arr):
    print("the array ")
    for x in arr:
        print(x, ' ')


def knn(test_point, samples, class_quant, k):
    dist_array = []
    sum_array = np.zeros(class_quant)
    lambda_array = np.zeros(len(samples))
    weights = np.ones(len(samples))
    for pt in samples:
        dist_array.append([ just_distance(test_point[0], test_point[1], pt[0], pt[1]), pt[2] ])
    dist_array = sorted(dist_array)
    print_array(dist_array)
    for c2 in range(k):
        sum_array[dist_array[c2][1]] += dist_array[c2][0]*weights[c2]
    print_array(sum_array)
    min_val = sum_array[0]
    for k in range(len(sum_array)):
        if sum_array[k] < min_val:
            min_val = sum_array[k]
            class_value = k
    return 'The test point is class '+str(class_value)


# Test point
tp = [10, 10]
CLASSES_QUANTITY = 2
k_neighbors = 7
L_C = 1
# [x,y,class]
data = [[1, 2, 0], [3, 4, 0], [5, 6, 1], [7, 7, 1], [2, 2, 0], [0, 2, 0], [7, 5, 1]]
print(knn(tp, data, CLASSES_QUANTITY, k_neighbors))
