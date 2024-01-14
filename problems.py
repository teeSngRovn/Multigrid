import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from multigrid_plan import Multigrid, RecGrid, Problem2D, Solution, Boundary, BoundaryType, RelaxationMethod

def SpecialError(f):
    return np.max(np.abs(f - 1))

def Unit(xx, yy):
    return 1

def Zero(xx, yy):
    return 0

def Func(xx,yy):
    return np.sin(10*np.pi*xx)*np.sin(10*np.pi*yy)

def PlotMatrix(matrix):
    # 指定颜色分辨率的边界
    bounds = np.linspace(-1, 1, 21)  # 0到1之间分成11份
    # 创建一个颜色映射对象，并绑定到数据范围  
    norm = mcolors.BoundaryNorm(bounds, plt.cm.Reds.N)
    # 使用 matshow 显示数据，并指定颜色映射和颜色分辨率  
    plt.matshow(matrix, cmap=plt.cm.Reds, norm=norm)
    # 添加 colorbar
    plt.colorbar() 
    plt.show()

def LaplaceMultigrid(grid:'RecGrid')->'Solution':
    '''
    用松弛方法在多重网格法下求解给定参数到指定误差的问题
    '''
    RelaxationMethods = RelaxationMethod()
    MultigridMethod = Multigrid(RelaxationMethods.GaussSeidel, grid = grid)
    # f代表当前迭代下的解, 生成初始解零构造
    MultigridMethod.f = MultigridMethod.grid.constructZeroMatrix()
    # boundary代表边界的矩阵表示
    MultigridMethod.boundary:'list[list[Boundary]]' = MultigridMethod.grid.constructBoundaryMatrix()
    # 给初值施加边界条件
    MultigridMethod.f = MultigridMethod.grid.ApplyBoundaryMatrix(matrix = MultigridMethod.f, boundaryMatrix=MultigridMethod.boundary)

    err = 1
    MultigridMethod.restriction()
    while err >= 0.001:
        _, err = MultigridMethod.Iteration(Zero)
    MultigridMethod.prolongation()
    # while err >= 0.001:
    #     _, err = MultigridMethod.Iteration(Zero)

    solution = Solution(MultigridMethod.f)
    print(MultigridMethod.errHistory)
    PlotMatrix(MultigridMethod.f)
        
    return solution

def LaplaceMultigridGaussFinite(grid:'RecGrid')->'Solution':
    '''
    用松弛方法在多重网格法下迭代指定次数求解给定参数的问题
    '''
    RelaxationMethods = RelaxationMethod()
    MultigridMethod = Multigrid(RelaxationMethods.GaussSeidel, grid = grid)
    # f代表当前迭代下的解, 生成初始解零构造
    MultigridMethod.f = MultigridMethod.grid.constructZeroMatrix()
    # boundary代表边界的矩阵表示
    MultigridMethod.boundary:'list[list[Boundary]]' = MultigridMethod.grid.constructBoundaryMatrix()
    # 给初值施加边界条件
    MultigridMethod.f = MultigridMethod.grid.ApplyBoundaryMatrix(matrix = MultigridMethod.f, boundaryMatrix=MultigridMethod.boundary)

    err = 1
    MultigridMethod.restriction()
    for _ in range(10):
        _, err = MultigridMethod.Iteration(Zero)
    MultigridMethod.prolongation()

    solution = Solution(MultigridMethod.f)
    print(MultigridMethod.errHistory)
    PlotMatrix(MultigridMethod.f)
        
    return solution

def LaplaceMultigridJacobiFinite(grid:'RecGrid')->'Solution':
    '''
    用松弛方法在多重网格法下迭代指定次数求解给定参数的问题
    '''
    RelaxationMethods = RelaxationMethod()
    MultigridMethod = Multigrid(RelaxationMethods.Jacobi, grid = grid)
    # f代表当前迭代下的解, 生成初始解零构造
    MultigridMethod.f = MultigridMethod.grid.constructZeroMatrix()
    # boundary代表边界的矩阵表示
    MultigridMethod.boundary:'list[list[Boundary]]' = MultigridMethod.grid.constructBoundaryMatrix()
    # 给初值施加边界条件
    MultigridMethod.f = MultigridMethod.grid.ApplyBoundaryMatrix(matrix = MultigridMethod.f, boundaryMatrix=MultigridMethod.boundary)

    err = 1
    MultigridMethod.restriction()
    for _ in range(10):
        _, err = MultigridMethod.Iteration(Zero)
    MultigridMethod.prolongation()

    solution = Solution(MultigridMethod.f)
    print(MultigridMethod.errHistory)
    PlotMatrix(MultigridMethod.f)
        
    return solution

def PoissonMultigrid(grid:'RecGrid')->'Solution':
    '''
    用松弛方法在多重网格法下求解给定参数到指定误差的问题
    '''
    RelaxationMethods = RelaxationMethod()
    MultigridMethod = Multigrid(RelaxationMethods.GaussSeidel, grid = grid)
    # f代表当前迭代下的解, 生成初始解零构造
    MultigridMethod.f = MultigridMethod.grid.constructZeroMatrix()
    # boundary代表边界的矩阵表示
    MultigridMethod.boundary:'list[list[Boundary]]' = MultigridMethod.grid.constructBoundaryMatrix()
    # 给初值施加边界条件
    MultigridMethod.f = MultigridMethod.grid.ApplyBoundaryMatrix(matrix = MultigridMethod.f, boundaryMatrix=MultigridMethod.boundary)

    err = 1
    MultigridMethod.restriction()
    while err >= 0.001:
        _, err = MultigridMethod.Iteration(Unit)
    MultigridMethod.prolongation()
    # while err >= 0.001:
    #     _, err = MultigridMethod.Iteration(Zero)

    solution = Solution(MultigridMethod.f)
    print(MultigridMethod.errHistory)
    PlotMatrix(MultigridMethod.f)
        
    return solution

def ProblemTest():
    nlevel = 4
    xlower = 0
    xupper = np.pi
    ylower = 0
    yupper = np.pi
    recBoundary = {
        "upper": Boundary(type = BoundaryType.dirichlet, val = 1),
        "lower": Boundary(type = BoundaryType.dirichlet, val = 0),
        "lleft": Boundary(type = BoundaryType.neumann, val = [1,0]),
        "right": Boundary(type = BoundaryType.neumann, val = [-1,0])
    }
    squareGrid = RecGrid(nlevel = nlevel, xlower = xlower, xupper = xupper, ylower = ylower, yupper = yupper, boundaries = recBoundary)
    SolveMethod = LaplaceMultigrid
    problem = Problem2D(SolveMethod = SolveMethod, grid = squareGrid)
    solution = problem.solve()

def ProblemP():
    nlevel = 16
    xlower = 0
    xupper = np.pi
    ylower = 0
    yupper = np.pi
    recBoundary = {
        "upper": Boundary(type = BoundaryType.dirichlet, val = 0),
        "lower": Boundary(type = BoundaryType.dirichlet, val = 0),
        "lleft": Boundary(type = BoundaryType.dirichlet, val = 0),
        "right": Boundary(type = BoundaryType.dirichlet, val = 0)
    }
    squareGrid = RecGrid(nlevel = nlevel, xlower = xlower, xupper = xupper, ylower = ylower, yupper = yupper, boundaries = recBoundary)
    SolveMethod = PoissonMultigrid
    problem = Problem2D(SolveMethod = SolveMethod, grid = squareGrid)
    solution = problem.solve()

def Problem1():
    nlevel = 4
    xlower = 0
    xupper = np.pi
    ylower = 0
    yupper = np.pi
    recBoundary = {
        "upper": Boundary(type = BoundaryType.dirichlet, val = 1),
        "lower": Boundary(type = BoundaryType.dirichlet, val = 1),
        "lleft": Boundary(type = BoundaryType.dirichlet, val = 1),
        "right": Boundary(type = BoundaryType.dirichlet, val = 1)
    }
    squareGrid = RecGrid(nlevel = nlevel, xlower = xlower, xupper = xupper, ylower = ylower, yupper = yupper, boundaries = recBoundary)
    SolveMethod = LaplaceMultigrid
    problem = Problem2D(SolveMethod = SolveMethod, grid = squareGrid)
    solution = problem.solve()

def Problem3():
    nlevel = 4
    xlower = 0
    xupper = np.pi
    ylower = 0
    yupper = np.pi
    recBoundary = {
        "upper": Boundary(type = BoundaryType.dirichlet, val = 0.5),
        "lower": Boundary(type = BoundaryType.dirichlet, val = 1),
        "lleft": Boundary(type = BoundaryType.dirichlet, val = 1),
        "right": Boundary(type = BoundaryType.dirichlet, val = 1)
    }
    squareGrid = RecGrid(nlevel = nlevel, xlower = xlower, xupper = xupper, ylower = ylower, yupper = yupper, boundaries = recBoundary)
    SolveMethod = LaplaceMultigrid
    problem = Problem2D(SolveMethod = SolveMethod, grid = squareGrid)
    solution = problem.solve()

def Problem4():
    nlevel = 8
    xlower = 0
    xupper = np.pi
    ylower = 0
    yupper = np.pi
    recBoundary = {
        "upper": Boundary(type = BoundaryType.discrete, val = [1,0]),
        "lower": Boundary(type = BoundaryType.discrete, val = [1,0]),
        "lleft": Boundary(type = BoundaryType.discrete, val = [1,0]),
        "right": Boundary(type = BoundaryType.discrete, val = [1,0])
    }
    squareGrid = RecGrid(nlevel = nlevel, xlower = xlower, xupper = xupper, ylower = ylower, yupper = yupper, boundaries = recBoundary)
    SolveMethod = LaplaceMultigridJacobiFinite
    problem = Problem2D(SolveMethod = SolveMethod, grid = squareGrid)
    solution = problem.solve()

if __name__=="__main__":
    '''
    注: 这里的upper是从x,y方向看, x代表i行, y代表j列
    Example:
    →y(j) 0 - pi
    ↓x(i) 0 - pi
                            lleft
                        
                        [[1,2,3,4,5],
                         [6,7,8,9,0],
                lower    [1,2,3,4,5],   upper
                         [6,7,8,9,0],
                         [1,2,3,4,5]]       
                            
                            right
    '''
    ProblemP()