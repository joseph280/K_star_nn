
import math
import numpy as np
import tensorflow as tf
import random


def just_distance(pt1, pt2, l_c):
    distance = math.sqrt((pt2[1]-pt1[1])**2+(pt2[0]-pt1[0])**2)
    return l_c*math.sqrt(distance)


def just_ndim_distance(pt1, pt2, LC):
    distance = 0
    for j in range(len(pt1)):
        distance += (pt2[j]-pt1[j])**2
    return LC*math.sqrt(distance)


def summation_lambda(count, beta_arr):
    sum = 0
    for k1 in range(count):
        sum += beta_arr[k1]
    return sum


def summation_sqr_lambda(count, beta_arr):
    summ = 0
    for k1 in range(count):
        summ += beta_arr[k1]**2
    return summ


def alpha_array_calc(lambda_k, beta_array, k):
    alph_arr = []
    mean = 0
    for tt in range(k):
        alph_arr.append(lambda_k[tt] - beta_array[tt])
        mean += lambda_k[tt] - beta_array[tt]
    # print ('mean is '+str(mean))
    new_arr = [w / mean for w in alph_arr]
    return new_arr


def f_estimation(labels, alpha_arr):
    fx0 = 0
    for z in range(len(alpha_arr)):
        # print ('label ' + str(labels[z])+' alpha ' + str(alpha_arr[z]))

        fx0 += labels[z]*alpha_arr[z]
    return fx0


def print_array(arr):
    print("the array ")
    for x in arr:
        print(x, ' ')


def k_star_nn(test_point, samples, class_quant, l_c_val, accuracy):
    # Init
    dist_array = []
    beta_array = []
    # print(test_point)
    labels = []
    n = len(samples)-1
    samples = samples
    lambda_array = np.zeros(len(samples))
    # Algorithm
    for pt in samples:
        dist_array.append([ just_distance(test_point, pt, l_c_val), pt[2] ])

        # else:
        #     print(pt)
    dist_array = sorted(dist_array)
    for q in dist_array:
        beta_array.append(q[0])
        labels.append(q[1])
    # print('dist array 0 ')
    # print_array(dist_array)

    # Lambda vector calculated
    lambda_array[0] = 1 + beta_array[0]
    k = 0
    # print('lambda ')
    # print(lambda_array[0])
    # print ('lambda_array[k] ' + str(lambda_array[k]) + 'beta array ' + str(beta_array[k]) + ' n ' + str(n))
    while lambda_array[k] > beta_array[k] and k <= n-1:
        k += 1

        inner_sqrt = math.sqrt(k + summation_lambda(k, beta_array)**2 - k * summation_sqr_lambda(k, beta_array) )
        # print ('inner ' +str(inner_sqrt)+' summation '+str(summation_lambda(k, beta_array))+' k '+str(k))
        lambda_array[k] = (1/k) * ( summation_lambda(k, beta_array) + inner_sqrt )
        # print ('lambda_array[k] ' + str(lambda_array[k]) + 'beta array ' + str(beta_array[k]) + ' n ' + str(n))

    # Alpha vector calculated
    alpha_array = alpha_array_calc(lambda_array, beta_array, k)

    # print ("Lambda")
    # print_array(lambda_array)
    # print ("Beta")
    # print_array(beta_array)
    # print ("Alphaa")
    # print_array(alpha_array)

    res = f_estimation(labels, alpha_array)

    if test_point[2] == int(round(res, 0)):
        accuracy += 1
    return accuracy
    # return 'The test point is class '+str(int(round(res, 0)))


def k_star_nn_cycle(samples, class_quant, l_c_val):
    acc = 0
    for point in samples:
        # Code to remove the test point from the samples

        samples_copy = samples
        for w in range(len(samples_copy)):
            if (point == samples_copy[w]).all():
                samples_copy = np.delete(samples_copy, w, 0)
                break
        # End of code to remove pt
        acc = k_star_nn(point, samples_copy, class_quant, l_c_val, acc)
    print('The accuracy of k*nn algorithm is: '+str((acc*100)/len(samples)))

# Test point
# tp = [10, 10]
# CLASSES_QUANTITY = 2
# L_C = 1
# # [x,y,class]
# data = [[1, 2, 0], [3, 4, 0], [5, 6, 1], [7, 7, 1], [2, 2, 0], [0, 2, 0], [7, 5, 1], [9, 8, 1], [9, 9, 1]]
# test_d = random.sample(data, int((50/100)*len(data)))
#
# print(test_d)
# print('************')
# train_d = list(filter(lambda x: x not in test_d, data))
# print(train_d)
# # print(knn(tp, data, CLASSES_QUANTITY, L_C))
