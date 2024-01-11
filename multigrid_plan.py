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
    def __init__(self, x:float, y:float, n0:int = 0, n1:int = 0):
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
    def __init__(self, nlevel:int, xlower:float, xupper:float, ylower:float, yupper:float, boundaries:'dict[str, Boundary]'):
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

    ''' 一个废掉的函数, 改为了更好的形式来写, 直接返回边界
    def constructBoundaryMatrix(self)->'tuple[list[list[float]],list[list[BoundaryType]]]':
        # 由矩形边界给出一个当前level的边界矩阵
        edgeList = [self.upper, self.lower, self.lleft, self.right]
        boundaryVal:list[list[float]] = [[0.0]*(self.nlevel+1)]*(self.nlevel+1)
        boundaryType:list[list[BoundaryType]] = [[BoundaryType.none]*(self.nlevel+1)]*(self.nlevel+1)
        boundary:list[list[Boundary]] = [[Boundary(BoundaryType.none, 0)]*(self.nlevel+1)]*(self.nlevel+1)
        for edge in edgeList:
            p1 = edge.p1
            p2 = edge.p2
            n0 = p1.n0
            n1 = p2.n1
            if (n0 == p2.n0):
                for i in range(self.nlevel + 1):
                    boundaryVal[n0][i] = edge.boundary.val
                    boundaryType[n0][i] = edge.boundary.type
                    boundary[n0][i] = edge.boundary
            else:
                for i in range(self.nlevel + 1):
                    boundaryVal[i][n1] = edge.boundary.val
                    boundaryType[i][n1] = edge.boundary.type
                    boundary[i][n1] = edge.boundary

        return boundaryVal, boundaryType
    '''
    
    def constructBoundaryMatrix(self)->'list[list[Boundary]]':
        '''
        由矩形边界给出一个当前level的边界矩阵
        '''
        edgeList = [self.upper, self.lower, self.lleft, self.right]
        boundary:list[list[Boundary]] = [[Boundary(BoundaryType.none, 0)]*(self.nlevel+1)]*(self.nlevel+1)
        for edge in edgeList:
            p1 = edge.p1
            p2 = edge.p2
            n0 = p1.n0
            n1 = p2.n1
            if (n0 == p2.n0):
                for i in range(self.nlevel + 1):
                    boundary[n0][i] = Boundary(type = edge.boundary.type, val = edge.boundary.val)
            else:
                for i in range(self.nlevel + 1):
                    boundary[i][n1] = Boundary(type = edge.boundary.type, val = edge.boundary.val)

        return boundary

                

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
    def __init__(self, iterMethod:'function', level0:int = 4):
        # f代表当前迭代下的解
        self.grid:'RecGrid'
        self.f:'np.array'
        # boundary代表边界的矩阵表示
        self.boundary:list[list[Boundary]]

        # iterMethod为当前采用的松弛迭代法
        self.iterMethod = iterMethod

        # iternum当前迭代步数
        self.iternum:int = 0

        # 当前level网格粗细程度
        self.level:int = level0

        # LevelHistory粗细程度的历史
        self.LevelHistory = [level0]

        # 迭代解的历史
        self.fHistory:list['np.array'] = list()
    
    def solve(self, f0:'function', grid:'RecGrid')->'Solution':
        '''
        用松弛方法在多重网格法下求解给定参数的问题
        '''
        # grid当前的网格
        self.grid:'RecGrid' = grid.constructGrid(self.level)
        
        # f代表当前迭代下的解
        self.f = self.grid.constructMatrixFromFunction(f0)
        
        # boundary代表边界的矩阵表示
        self.boundary = self.grid.constructBoundaryMatrix()

        for i in range(3):
            self.restriction()
            self.Iteration()
            self.prolongation()

        solution = Solution(self.f)
        
        return solution


    def getNLevel(self)->int:
        '''
        获取当前的网格粗细程度
        '''
        return self.level

    def Iteration(self, boundary:'function'):
        '''
        传入一个迭代的方法和边界, 用该方法迭代一步
        method : 要使用的松弛迭代方法
        boundary : 边界条件
        '''
        self.f = self.iterMethod(self.f, boundary = boundary)
        self.fHistory.append(self.f)
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


class RelaxationMethod:
    def GaussSeidel(self, f: 'np.array', boundary:'list[list[Boundary]]'):
        '''
        迭代求解一步
        f : 当前的解
        grid : 所构造的网格
        '''
        error:float = 0
        sizeY = len(boundary)
        sizeX = len(boundary[0])
        for i in range(sizeX):
            for j in range(sizeY):
                if (boundary[i][j].type == BoundaryType.none):
                    val = (f[i-1][j] + f[i+1][j] + f[i][j-1]+f[i][j+1])/4
                    dval = f[i][j] - val
                    error = np.max(error, np.abs(dval))
                    f[i][j] = val
        
        return f


class Solution:
    def __init__(self, anything):
        self.result = anything


class Problem2D:
    def __init__(self, f0: 'function', method:'Multigrid', grid:'RecGrid'):
        self.func = f0
        self.grid = grid
        self.method = method
        self.solution = Solution()

    def solve(self)->"Solution":
        self.solution = self.method.solve(self.f0, self.grid)
        return self.solution


def Func(xx, yy):
    return np.sin(xx)*np.cos(yy)

if __name__=="__main__":
    x0 = 0
    x1 = np.pi
    y0 = 0
    y1 = np.pi
    recBoundary = {
        "upper": Boundary(type = BoundaryType.diriclet),
        "lower": Boundary(type = BoundaryType.diriclet),
        "lleft": Boundary(type = BoundaryType.diriclet),
        "right": Boundary(type = BoundaryType.diriclet)
    }
    squareGrid = RecGrid(nlevel = 4, xlower = x0, xupper = x1, ylower = y0, yupper = y1, boundaries = recBoundary)
    RelaxationMethods = RelaxationMethod()
    MultigridMethod = Multigrid(RelaxationMethods.GaussSeidel, level0 = 4)
    problem = Problem2D(f0 = Func, method = MultigridMethod, grid = squareGrid)
    solution = problem.solve()