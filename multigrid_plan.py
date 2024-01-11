import numpy as np
from enum import Enum

class BoundaryType(Enum):
    none = 0
    diriclet = 1
    neumann = 2

class Boundary:
    def __init__(self, type:BoundaryType = BoundaryType.dirichlet, val:float = 0):
        '''
        type : 边界类型
        val : 边界值
        '''
        self.type = type
        self.val = val

    def change2dirichlet(self, val:float):
        self.type = BoundaryType.diriclet
        self.val = val

    def change2neumann(self, val:float):
        self.type = BoundaryType.neumann
        self.val = val

class Point:
    def __init__(self, x, y, n0 = 0, n1 = 0):
        self.x = x
        self.y = y
        self.n0 = n0
        self.n1 = n1

class Line:
    def __init__(self, p1:Point, p2:Point, boundary:Boundary):
        self.boundary = boundary
        self.p1 = p1
        self.p2 = p2

class RecGrid:
    def __init__(self, nlevel, xlower, xupper, ylower, yupper, boundaries:'dict[str, Boundary]'):
        '''
        构造网格, nlevel代表当前网格的粗细程度
        '''
        # nlevel网格粗细程度
        self.nlevel:int = nlevel
        
        # 矩形的点
        self.ld = Point(xlower, ylower, n0 = 0, n1 = 0)
        self.rd = Point(xupper, ylower, n0 = nlevel - 1, n1 = 0)
        self.lu = Point(xlower, yupper, n0 = 0, n1 = nlevel - 1)
        self.ru = Point(xupper, yupper, n0 = nlevel - 1, n1 = nlevel - 1)
        
        # 矩形的边
        self.upper = Line(self.lu, self.ru, boundaries["upper"])
        self.lower = Line(self.ld, self.rd, boundaries["lower"])
        self.lleft = Line(self.lu, self.ld, boundaries["lleft"])
        self.right = Line(self.ru, self.rd, boundaries["right"])

    def constructGrid(self, nlevel) -> 'RecGrid':
        '''
        构造网格, nlevel代表当前网格的粗细程度
        '''
        if nlevel == self.nlevel: return self
        result = RecGrid(nlevel, self.xlower, self.xupper, self.ylower, self.yupper, self.boundaries)
        return result


    def constructBoundaryMatrix(self)->'tuple[list[list[float]],list[list[BoundaryType]]]':
        '''
        由矩形边界给出一个当前level的边界矩阵
        '''
        edgeList = [self.upper, self.lower, self.lleft, self.right]
        boundaryType = [[BoundaryType.none]*(self.nlevel+1)]*(self.nlevel+1)
        boundaryVal = [[0.0]*(self.nlevel+1)]*(self.nlevel+1)
        for edge in edgeList:
            p1 = edge.p1
            p2 = edge.p2
            n0 = p1.n0
            n1 = p2.n1
            if (n0 == p2.n0):
                for i in range(self.nlevel + 1):
                    boundaryVal[n0][i] = edge.boundary.val
                    boundaryType[n0][i] = edge.boundary.type
            else:
                for i in range(self.nlevel + 1):
                    boundaryVal[i][n1] = edge.boundary.val
                    boundaryType[i][n1] = edge.boundary.type

        return boundaryVal, boundaryType

                

    def constructSolutionMatrixFromFunction(self, f0:'function')->'np.array':
        '''
        由函数f0得到当前grid上生成矩阵
        '''
        x = np.linspace(self.xlower, self.xupper, self.nlevel + 1)
        y = np.linspace(self.ylower, self.yupper, self.nlevel + 1)
        xx, yy = np.meshgrid(x, y, indexing="ij")
        f = f0(xx, yy)


        
        return f

    def PlotGrid(self):
        '''
        绘制网格点
        '''
        pass
    
class Multigrid:
    def __init__(self, f0:'function', level0:int = 4):
        # grid当前的网格
        self.grid:'RecGrid' = RecGrid(level0)
        
        # f代表当前迭代下的解
        self.f = self.grid.constructMatrixFromFunction(f0)

        # boundary代表边界的矩阵表示
        self.boundaryVal, self.boundaryType = self.grid.constructBoundaryMatrix()

        # iternum当前迭代步数
        self.iternum:int = 0

        # 当前level网格粗细程度
        self.level:int = level0

        # LevelHistory粗细程度的历史
        self.LevelHistory = [level0]

        # 迭代解的历史
        self.fHistory = [self.f]

    def Iteration(self, method:'function', boundary:'function'):
        '''
        传入一个迭代的方法和边界, 用该方法迭代一步
        method : 要使用的松弛迭代方法
        boundary : 边界条件
        '''
        self.fHistory.append(self.f)
        self.f = method(self.f, boundary = boundary)
        self.iternum += 1
        return self.f

    def restriction(self):
        '''
        从nlevel大的网格到nlevel小的网格, 默认2n -> n;
        '''
        pass

    def prolongation(self):
        '''
        从nlevel小的网格到nlevel大的网格, 默认n -> 2n;
        '''
        pass

    def getNLevel(self):
        '''
        获取当前的网格粗细程度
        '''
        return self.level


class RelaxationMethod:
    def GaussSeidel(self, f: 'np.array', grid:'RecGrid'):
        '''
        迭代求解一步
        f : 当前的解
        grid : 所构造的网格
        '''
        pass

def Func(xx, yy):
    return np.sin(xx)*np.cos(yy)

methods = RelaxationMethod()