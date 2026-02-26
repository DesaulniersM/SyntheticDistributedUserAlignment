import numpy as np

np.set_printoptions(precision = 5, suppress=True)

class NoiseInjector:
    def __init__(self, sdxy, sdz):
        self.sdxy = sdxy # standard deviation for the xy plane
        self.sdz = sdz # standard deviation for the z distance

    # def transform(self, points, matrix):
        # self.points = points @ matrix.T # applies the transformation matrix to the noise-added points
        # return self.points
    
    def add_noise(self, points):
        points = np.array(points, dtype = float)

        if points.ndim == 1: # checks if the point array is one dimensional
            points = points.reshape(1,-1) # makes the point array two dimensional
        
        # x and y noise
        xyNoise = np.random.normal(loc = 0, scale = self.sdxy, size = (points.shape[0], 2)) 
        points[:, :2] += xyNoise

        # z noise
        zNoise = np.random.normal(loc = 0, scale = self.sdz, size = points.shape[0]) 
        points[:, 2] += zNoise 

        # self.points = points

        return points











        
        



 




