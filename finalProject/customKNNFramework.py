import numpy as np
from sklearn.neighbors import BallTree
from typing import Callable
from sklearn.preprocessing import StandardScaler

features_data = np.array([[1, 2, 3], [1, 2, 4], [5, 5, 5]])#On and on, assumes 3 features
class_data = np.array([1, 1, 0])#Class of 1 or 0, corresponds to same index data in features_data
scaler = StandardScaler()
features_data = scaler.fit_transform(features_data)
#Make search-tree
tree = BallTree(features_data)
def weighted_knn_framework(weighter:Callable, n:int, point:np.ndarray):
    #Testing point
    #a_point = np.array([0.5, 0.5, 0.5])
    point = scaler.transform(np.array([point]))[0]
    #Weighter needs a boolean value .handles0distance

    distances, indices = tree.query(point.reshape(1, -1), k=n)#Gets closest n points

    #Get those data points
    closest_n_points_features = features_data[indices[0]]
    closest_n_points_classes = class_data[indices[0]]
    class_choices = list(set(closest_n_points_classes))
    for choice in class_choices: assert isinstance(choice, (int, np.integer))#So, all classes are integers!
    class_choices_weights = np.zeros(len(class_choices))
    match_use = None
    if getattr(weighter, "handles0distance", False):
        for newpoint, point_class, distance in zip(closest_n_points_features, closest_n_points_classes, distances[0]):
            class_index = class_choices.index(point_class)
            class_choices_weights[class_index] += weighter(point, newpoint, distance)#Custom weighter function decides how much to add for any given point!
    else:
        for newpoint, point_class, distance in zip(closest_n_points_features, closest_n_points_classes, distances[0]):
            class_index = class_choices.index(point_class)
            if distance==0:
                match_use = point_class
            else:
                class_choices_weights[class_index] += weighter(point, newpoint, distance)#Custom weighter function decides how much to add for any given point!
    if match_use is not None:
        return match_use
    return class_choices[np.argmax(class_choices_weights)]




