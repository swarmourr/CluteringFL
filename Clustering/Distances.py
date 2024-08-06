from enum import Enum

class DistanceMetric(Enum):
    EUCLIDEAN = 'euclidean'
    COSINE = 'cosine'
    MANHATTAN = 'manhattan'
    MINKOWSKI = 'minkowski'
    JACCARD = 'jaccard'
