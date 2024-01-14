import numpy as np
from enum import Enum

class BoundaryType(Enum):
    none = 0
    dirichlet = 1
    neumann = 2
    discrete = 3

class Boundary:
    def __init__(self, type:BoundaryType = BoundaryType.dirichlet, val = 0):
        '''
        type : 边界类型
        val : 边界值
        '''
        self.type = type
        self.val = val

    def change2dirichlet(self, val:float):
        self.type = BoundaryType.dirichlet
        self.val = val

    def change2neumann(self, val):
        self.type = BoundaryType.neumann
        self.val = val

    def change2discrete(self, val:int):
        '''
        val : 何种粗糙程度定义的
        '''
        self.type = BoundaryType.discrete
        self.val = val

class Point:
    def __init__(self, x:float, y:float, n0:int = 0, n1:int = 0):
        # coordinate:
        self.x = x
        self.y = y
        # index:
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
        self.rd = Point(xupper, ylower, n0 = nlevel, n1 = 0)
        self.lu = Point(xlower, yupper, n0 = 0, n1 = nlevel)
        self.ru = Point(xupper, yupper, n0 = nlevel, n1 = nlevel)
        
        # 矩形的边
        self.upper = Line(self.lu, self.ru, boundaries["upper"])
        self.lower = Line(self.ld, self.rd, boundaries["lower"])
        self.lleft = Line(self.lu, self.ld, boundaries["lleft"])
        self.right = Line(self.ru, self.rd, boundaries["right"])

        # 存一下变量
        self.boundaries = boundaries
        self.dx = (xupper - xlower) / nlevel
        self.dy = (yupper - ylower) / nlevel
        self.ds = self.dx * self.dy

    def constructGrid(self, nlevel) -> 'RecGrid':
        '''
        构造网格, nlevel代表当前网格的粗细程度
        '''
        if nlevel == self.nlevel: return self
        result = RecGrid(nlevel, self.ld.x, self.rd.x, self.ld.y, self.lu.y, self.boundaries)
        return result
       
    def constructBoundaryMatrix(self)->'list[list[Boundary]]':
        '''
        由矩形边界给出一个当前level的边界矩阵
        '''
        edgeList = [self.upper, self.lower, self.lleft, self.right]
        boundary:list[list[Boundary]] = [[Boundary(BoundaryType.none, 0) for _ in range(self.nlevel+1)]for _ in range(self.nlevel+1)]
        for edge in edgeList:
            p1 = edge.p1
            p2 = edge.p2
            n0 = p1.n0
            n1 = p2.n1
            if (n0 == p2.n0):
                # 左右边界
                for i in range(self.nlevel + 1):
                    # 更希望他是dirichlet的边界而非其它的的边界
                    if boundary[n0][i].type == BoundaryType.dirichlet: continue    
                    boundary[n0][i] = Boundary(type = edge.boundary.type, val = edge.boundary.val)
            else:
                # 上下边界
                for i in range(self.nlevel + 1):
                    if boundary[i][n1].type == BoundaryType.dirichlet: continue
                    boundary[i][n1] = Boundary(type = edge.boundary.type, val = edge.boundary.val)

        return boundary

    def constructMatrixFromFunction(self, f0:'function')->'list[list[float]]':
        '''
        由函数f0得到当前grid上生成矩阵
        '''
        x = np.linspace(self.ld.x, self.rd.x, self.nlevel + 1)
        y = np.linspace(self.ld.y, self.lu.y, self.nlevel + 1)
        sizeX = len(x)
        sizeY = len(y)
        f = [[0 for _ in range(sizeY)]for _ in range(sizeX)]
        for i in range(sizeX):
            for j in range(sizeY):
                xx = x[i]
                yy = y[i]
                f[i][j] = f0(xx,yy)
        
        return f
    
    def constructZeroMatrix(self)->'np.array':
        '''
        由函数f0得到当前grid上生成矩阵
        '''
        f = np.zeros((self.nlevel + 1, self.nlevel + 1))
        
        return f
    
    def ApplyBoundaryMatrix(self, matrix:'np.array', boundaryMatrix:'list[list[Boundary]]')->'np.array':
        '''
        通过边界矩阵给当前解施加边界条件
        '''
        sizeY = matrix.shape[0]
        sizeX = matrix.shape[1]
        for i in range(sizeX):
            for j in range(sizeY):
                # 不同的边界条件的赋值处理
                if (boundaryMatrix[i][j].type == BoundaryType.dirichlet): matrix[i][j] = boundaryMatrix[i][j].val
                elif (boundaryMatrix[i][j].type == BoundaryType.discrete):
                    if ((i + j) % 2 == 0):
                        matrix[i][j] = boundaryMatrix[i][j].val[0]
                    else:
                        matrix[i][j] = boundaryMatrix[i][j].val[1]
                else:
                    continue
        return matrix

class Multigrid:
    def __init__(self, iterMethod:'function', grid:'RecGrid'):
        # grid代表当前粗糙程度的网格
        self.grid:'RecGrid' = grid
        
        # f代表当前迭代下的解
        self.f:'np.array'
        
        # boundary代表边界的矩阵表示
        self.boundary:list[list[Boundary]]

        # iterMethod为当前采用的松弛迭代法
        self.iterMethod = iterMethod

        # iternum当前迭代步数
        self.iternum:int = 0

        # 当前level网格粗细程度
        self.level:int = grid.nlevel

        # LevelHistory粗细程度的历史
        self.LevelHistory = [self.level]

        # 迭代解的历史
        self.fHistory:list['np.array'] = list()
        
        # 当前迭代下的误差
        self.err:float = 0

        # 迭代误差的历史
        self.errHistory:list['float'] = list()

    def getNLevel(self)->int:
        '''
        获取当前的网格粗细程度
        '''
        return self.level

    def Iteration(self, f0:'function')->"tuple[np.array, float]":
        '''
        传入一个迭代的方法和边界, 用该方法迭代一步
        f0: 可能存在的右侧函数
        '''
        f0 = self.grid.constructMatrixFromFunction(f0)
        self.f, self.err = self.iterMethod(self.f, boundary = self.boundary, f0 = f0, ds=self.grid.ds)
        self.fHistory.append(self.f)
        self.errHistory.append(self.err)
        return self.f, self.err

    def restriction(self):
        '''
        从nlevel大的网格到nlevel小的网格, 默认2n -> n;
        注意在改动解f时boundary也是要变动的
        使用全加权方法
        '''
        self.level = self.level // 2
        self.LevelHistory.append(self.level)
        self.grid = self.grid.constructGrid(self.level)
        self.boundary = self.grid.constructBoundaryMatrix()
        sizeX = len(self.boundary)
        sizeY = len(self.boundary[0])
        f = np.zeros((sizeX, sizeY))
        for i in range(sizeX):
            for j in range(sizeY):
                if i == 0 or j == 0 or i == sizeX - 1 or j == sizeY - 1: 
                    # 边界，为了迎合Neumann边界条件
                    f[i][j] = self.f[i * 2][j * 2]
                else:
                    # 内点
                    # i,j是粗网格的坐标，cx,cy是细网格的坐标
                    cx = i * 2
                    cy = j * 2
                    f[i][j] = 1/4*(
                              self.f[cx][cy]
                          ) + 1/8*(
                              self.f[cx - 1][cy] + self.f[cx + 1][cy] + self.f[cx][cy - 1] + self.f[cx][cy + 1]
                          ) + 1/16*(
                              self.f[cx - 1][cy - 1] + self.f[cx + 1][cy - 1] + self.f[cx - 1][cy + 1] + self.f[cx + 1][cy + 1]
                          )
        
        self.f = self.grid.ApplyBoundaryMatrix(matrix = f, boundaryMatrix=self.boundary)


    def prolongation(self):
        '''
        从nlevel小的网格到nlevel大的网格, 默认n -> 2n;
        注意在改动解f时boundary也是要变动的
        '''
        self.level = self.level * 2
        self.LevelHistory.append(self.level)
        self.grid = self.grid.constructGrid(self.level)
        self.boundary = self.grid.constructBoundaryMatrix()
        sizeX = len(self.boundary)
        sizeY = len(self.boundary[0])
        f = np.zeros((sizeX, sizeY))
        for i in range(sizeX):
            for j in range(sizeY):
                # i,j是细网格的坐标
                if i % 2 == 0 and j % 2 == 0: 
                    # 与原坐标点重合
                    f[i][j] = self.f[i//2][j//2]
                elif i % 2 == 0 and j % 2 == 1:
                    # 与原坐标点在x方向重合
                    f[i][j] = (self.f[i//2][j//2] + self.f[i//2][j//2 + 1])/2
                elif i % 2 == 1 and j % 2 == 0:
                    # 与原坐标点在y方向重合
                    f[i][j] = (self.f[i//2][j//2] + self.f[i//2 + 1][j//2])/2
                else:
                    # 与原坐标点不重合
                    f[i][j] = (self.f[i//2][j//2] + self.f[i//2 + 1][j//2] + self.f[i//2][j//2 + 1] + self.f[i//2 + 1][j//2 + 1])/4
        
        self.f = self.grid.ApplyBoundaryMatrix(matrix = f, boundaryMatrix=self.boundary)


class RelaxationMethod:
    def Jacobi(self, f: 'np.array', boundary:'tuple[list[list[Boundary]], float]', f0:'np.array', ds:float = 0.1):
        '''
        迭代求解一步
        f : 当前的解
        f0 : 泊松方程的矩阵
        grid : 所构造的网格
        '''
        tmp_f = f.copy()
        error:float = 0.0
        sizeX = len(boundary)
        sizeY = len(boundary[0])
        for i in range(sizeX):
            for j in range(sizeY):
                if (boundary[i][j].type == BoundaryType.dirichlet): continue
                if (boundary[i][j].type == BoundaryType.discrete): continue
                if (boundary[i][j].type == BoundaryType.neumann):
                    offsetI = boundary[i][j].val[0]
                    offsetJ = boundary[i][j].val[1]
                    val = tmp_f[i+offsetI][j+offsetJ]
                    f[i][j] = val
                if (boundary[i][j].type == BoundaryType.none):
                    val = (tmp_f[i-1][j] + tmp_f[i+1][j] + tmp_f[i][j-1]+tmp_f[i][j+1] - (ds**2)*f0[i][j])/4
                    dval = tmp_f[i][j] - val
                    error = max(error, abs(dval))
                    f[i][j] = val
        
        return f, error
    
    def GaussSeidel(self, f: 'np.array', boundary:'tuple[list[list[Boundary]], float]', f0:'np.array', ds:float = 0.1):
        '''
        迭代求解一步
        f : 当前的解
        f0 : 泊松方程的矩阵
        grid : 所构造的网格
        '''
        error:float = 0.0
        sizeX = len(boundary)
        sizeY = len(boundary[0])
        for i in range(sizeX):
            for j in range(sizeY):
                if (boundary[i][j].type == BoundaryType.dirichlet): continue
                if (boundary[i][j].type == BoundaryType.discrete): continue
                if (boundary[i][j].type == BoundaryType.neumann):
                    offsetI = boundary[i][j].val[0]
                    offsetJ = boundary[i][j].val[1]
                    val = f[i+offsetI][j+offsetJ]
                    f[i][j] = val
                if (boundary[i][j].type == BoundaryType.none):
                    val = (f[i-1][j] + f[i+1][j] + f[i][j-1]+f[i][j+1] - (ds**2)*f0[i][j])/4
                    dval = f[i][j] - val
                    error = max(error, abs(dval))
                    f[i][j] = val
        
        return f, error


class Solution:
    def __init__(self, anything):
        self.result = anything


class Problem2D:
    def __init__(self, SolveMethod:'function', grid:'RecGrid'):
        self.grid = grid
        self.method = SolveMethod
        self.solution = Solution(None)

    def solve(self)->"Solution":
        self.solution = self.method(self.grid)
        return self.solution