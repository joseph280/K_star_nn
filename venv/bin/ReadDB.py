# Load CSV using Pandas from URL
import pandas
import random
from KnnAdaptativeNeighborsWeightsAlgorithm import k_star_nn_cycle
from BasicKnnAlgorithm import knn_cycle
from BasicKnnAlgorithm import knn_sklearn_cycle
from WeightedKnnAlgorithm import k_weighted_nn_cycle


def print_array(arr):
    print("the array ")
    for x in arr:
        print(x, ' ')


def divide_train_test(values, percentage):
    train_d = random.sample(values, int((percentage/100)*len(values)))
    test_d = list(filter(lambda x: x not in train_d, data))
    return train_d, test_d


# DB 1

print('***************************************************************')
print('The results for the "diabetes" DB are:')

url = "/media/Data/DBs_ML/Diabetes/diabetes.csv"
# url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
random.seed()
# names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pandas.read_csv(url)
# data = pandas.read_csv(url, names=names, sep=';')
# print(data.shape)
# print(data.head())
# print(data.columns.values)
# print(data.index)
# print(data.values)
# db1 = data[['Insulin', 'DiabetesPedigreeFunction', 'Outcome']]
# db1 = data[['Glucose', 'DiabetesPedigreeFunction', 'Outcome']]
db1 = data[['Glucose', 'Age', 'Outcome']]
# print('.......**********')
# print(db)
# print(len(labelsArr))
# print('............')
# print(data.loc[5])
# print('............')
# print(data.Glucose)

# print('----------')
array_db1 = db1.values
# for pt in array_db:
#     print(pt[0],' * ',pt[1],' * ',pt[2])
# print(array_db[0])

# tp = [148, 50]
# Perc = 70
CLASSES_QUANTITY = 2
L_C = 0.3
basic_k = 3
weighted_k = 10

knn_sklearn_cycle(array_db1, CLASSES_QUANTITY, basic_k)
knn_cycle(array_db1, CLASSES_QUANTITY, basic_k)
k_weighted_nn_cycle(array_db1, CLASSES_QUANTITY, weighted_k)
k_star_nn_cycle(array_db1, CLASSES_QUANTITY, L_C)

# DB 2

print('***************************************************************')
print('The results for the "biodegradation" DB are:')

# tp = [148, 50]
# Perc = 70
CLASSES_QUANTITY = 2
L_C = 2
basic_k = 3
weighted_k = 10


url2 = "/media/Data/DBs_ML/QSAR/biodeg.csv"
#
names2 = range(42)
data2 = pandas.read_csv(url2, names=names2, sep=',')
# data2 = pandas.read_csv(url2)
# print(data2.columns.values)
# labelsArr = list(data['Outcome'])
# Number of heavy atoms  is [2]  and
db2 = data2[[12, 27, 41]]
array_db2 = db2.values
# print(data2)
# print(db2)
knn_sklearn_cycle(array_db2, CLASSES_QUANTITY, basic_k)
knn_cycle(array_db2, CLASSES_QUANTITY, basic_k)
k_weighted_nn_cycle(array_db2, CLASSES_QUANTITY, weighted_k)
k_star_nn_cycle(array_db2, CLASSES_QUANTITY, L_C)

# DB 3

print('***************************************************************')
print('The results for the "ionosphere" DB are:')

# tp = [148, 50]
# Perc = 70
CLASSES_QUANTITY = 2
L_C = 1
basic_k = 3
weighted_k = 5


url3 = "/media/Data/DBs_ML/Ionosphere/ionosphere.csv"

names3 = range(35)
data3 = pandas.read_csv(url3, names=names3, sep=',')
# data2 = pandas.read_csv(url2)
# print(data2.columns.values)
# labelsArr = list(data['Outcome'])
# Number of heavy atoms  is [2]  and
db3 = data3[[0, 7, 34]]
array_db3 = db3.values

knn_sklearn_cycle(array_db3, CLASSES_QUANTITY, basic_k)
knn_cycle(array_db3, CLASSES_QUANTITY, basic_k)
k_weighted_nn_cycle(array_db3, CLASSES_QUANTITY, weighted_k)
k_star_nn_cycle(array_db3, CLASSES_QUANTITY, L_C)




