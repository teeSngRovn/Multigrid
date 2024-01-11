import numpy as np
import matplotlib.pyplot as plt

class Grid:
    def __init__(self, range:iter, dimension = 2, nlevel = 0):
        '''
        range: should be a iteration object at specific dimension, for example dimension = 2 should be look like[[xmin, xmax],[ymin, ymax]]
        '''
        self.dimension = dimension
        self.qmin = [xrange[0] for xrange in range]
        self.qmax = [xrange[1] for xrange in range]
        self.nlevel = nlevel
        self.interval = 2 ** nlevel
        self.sitesNumber = 2 ** nlevel + 1
        self.grid = self.constructGrid()
        self.solution = self.constructSolution()

    def Update(self, range:iter = None, dimension = None, nlevel = None):
        if dimension != None:
            self.dimension = dimension
        if range != None: 
            self.qmin = [xrange[0] for xrange in range]
            self.qmax = [xrange[1] for xrange in range]
        if nlevel != None: 
            self.nlevel = nlevel
            self.interval = 2 ** nlevel
            self.sitesNumber = 2 ** nlevel + 1
        if not(dimension==None and range == None and nlevel == None):
            self.grid = self.construct()
    
    def constructGrid(self):
        dimension = self.dimension
        qmin = self.qmin
        qmax = self.qmax
        sitesNumber = self.sitesNumber
        axis = list()
        for i in range(dimension):
            ax = np.linspace(qmin[i], qmax[i], sitesNumber)
            axis.append(ax)

        grid = np.meshgrid(axis ,indexing="ij")
        
        return grid

    def constructSolutionGrid(self):
        sitesNumber = self.sitesNumber
        dimension = self.dimension
        sizes = list()
        for i in range(dimension):
            sizes.append(sitesNumber)
        solution = np.zeros(sizes)
        
        return solution
        
    def findNearest(self, x: float, y:float)-> 'np.array':
        pass

    def restriction(self, fromgrid:'Grid')-> 'np.array':
        fromgrid.findNearest(x,y)

    def prolongation(self, togrid:'Grid')-> 'np.array':
        pass

class RelaxationMethod:
    def GaussSiedel(self):
        pass

class MultipleGridSolver:
    def ApplyingMethod(self, grid:Grid, method:RelaxationMethod):
        pass


