import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from src.Equation2D import Multigrid, RecGrid, Problem2D, Solution, Boundary, BoundaryType, RelaxationMethod

def Zero(xx,yy):
    return 0

def PlotMatrix(matrix, id):
    # 指定颜色分辨率的边界
    bounds = np.linspace(-1, 1, 21)  # 0到1之间分成11份
    # 创建一个颜色映射对象，并绑定到数据范围  
    norm = mcolors.BoundaryNorm(bounds, plt.cm.Reds.N)
    # 使用 matshow 显示数据，并指定颜色映射和颜色分辨率  
    plt.matshow(matrix, cmap=plt.cm.Reds, norm=norm)
    # 添加 colorbar
    plt.colorbar()
    plt.savefig(f"{id}")
    plt.clf()
    # plt.show()

def LaplaceMultigrid(grid:'RecGrid')->'Solution':
    '''
    用松弛方法在多重网格法下求解给定参数到指定误差的问题
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
    num_steps = 0
    while err >= 0.000001:
        if (num_steps%5==0):PlotMatrix(MultigridMethod.f, num_steps)
        _, err = MultigridMethod.Iteration(Zero)
        num_steps += 1

    solution = Solution(MultigridMethod.f)
    print(MultigridMethod.errHistory)
    print("Number of steps: ", num_steps)
        
    return solution

def Problem():
    nlevel = 8
    xlower = 0;xupper = np.pi;ylower = 0;yupper = np.pi
    recBoundary = {
        "upper": Boundary(type = BoundaryType.dirichlet, val = 1),
        "lower": Boundary(type = BoundaryType.dirichlet, val = 1),
        "lleft": Boundary(type = BoundaryType.dirichlet, val = 1),
        "right": Boundary(type = BoundaryType.dirichlet, val = 0)
    }
    squareGrid = RecGrid(nlevel = nlevel, xlower = xlower, xupper = xupper, ylower = ylower, yupper = yupper, boundaries = recBoundary)
    SolveMethod = LaplaceMultigrid
    problem = Problem2D(SolveMethod = SolveMethod, grid = squareGrid)
    solution = problem.solve()

Problem()