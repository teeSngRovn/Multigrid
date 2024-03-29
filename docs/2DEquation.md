Help on module 2DEquation:

NAME
    2DEquation

CLASSES
    builtins.object
        Boundary
        Line
        Multigrid
        Point
        Problem2D
        RecGrid
        RelaxationMethod
        Solution
    enum.Enum(builtins.object)
        BoundaryType
    
    class Boundary(builtins.object)
     |  Boundary(type: 2DEquation.BoundaryType = <BoundaryType.dirichlet: 1>, val=0)
     |  
     |  Methods defined here:
     |  
     |  __init__(self, type: 2DEquation.BoundaryType = <BoundaryType.dirichlet: 1>, val=0)
     |      type : 边界类型
     |      val : 边界值
     |  
     |  change2dirichlet(self, val: float)
     |  
     |  change2discrete(self, val: int)
     |      val : 何种粗糙程度定义的
     |  
     |  change2neumann(self, val)
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)
    
    class BoundaryType(enum.Enum)
     |  BoundaryType(value, names=None, *, module=None, qualname=None, type=None, start=1)
     |  
     |  An enumeration.
     |  
     |  Method resolution order:
     |      BoundaryType
     |      enum.Enum
     |      builtins.object
     |  
     |  Data and other attributes defined here:
     |  
     |  dirichlet = <BoundaryType.dirichlet: 1>
     |  
     |  discrete = <BoundaryType.discrete: 3>
     |  
     |  neumann = <BoundaryType.neumann: 2>
     |  
     |  none = <BoundaryType.none: 0>
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors inherited from enum.Enum:
     |  
     |  name
     |      The name of the Enum member.
     |  
     |  value
     |      The value of the Enum member.
     |  
     |  ----------------------------------------------------------------------
     |  Readonly properties inherited from enum.EnumMeta:
     |  
     |  __members__
     |      Returns a mapping of member name->value.
     |      
     |      This mapping lists all enum members, including aliases. Note that this
     |      is a read-only view of the internal mapping.
    
    class Line(builtins.object)
     |  Line(p1: 2DEquation.Point, p2: 2DEquation.Point, boundary: 2DEquation.Boundary)
     |  
     |  Methods defined here:
     |  
     |  __init__(self, p1: 2DEquation.Point, p2: 2DEquation.Point, boundary: 2DEquation.Boundary)
     |      Initialize self.  See help(type(self)) for accurate signature.
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)
    
    class Multigrid(builtins.object)
     |  Multigrid(iterMethod: 'function', grid: 'RecGrid')
     |  
     |  Methods defined here:
     |  
     |  Iteration(self, f0: 'function') -> 'tuple[np.array, float]'
     |      传入一个迭代的方法和边界, 用该方法迭代一步
     |      f0: 可能存在的右侧函数
     |  
     |  __init__(self, iterMethod: 'function', grid: 'RecGrid')
     |      Initialize self.  See help(type(self)) for accurate signature.
     |  
     |  getNLevel(self) -> int
     |      获取当前的网格粗细程度
     |  
     |  prolongation(self)
     |      从nlevel小的网格到nlevel大的网格, 默认n -> 2n;
     |      注意在改动解f时boundary也是要变动的
     |  
     |  restriction(self)
     |      从nlevel大的网格到nlevel小的网格, 默认2n -> n;
     |      注意在改动解f时boundary也是要变动的
     |      使用全加权方法
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)
    
    class Point(builtins.object)
     |  Point(x: float, y: float, n0: int = 0, n1: int = 0)
     |  
     |  Methods defined here:
     |  
     |  __init__(self, x: float, y: float, n0: int = 0, n1: int = 0)
     |      Initialize self.  See help(type(self)) for accurate signature.
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)
    
    class Problem2D(builtins.object)
     |  Problem2D(SolveMethod: 'function', grid: 'RecGrid')
     |  
     |  Methods defined here:
     |  
     |  __init__(self, SolveMethod: 'function', grid: 'RecGrid')
     |      Initialize self.  See help(type(self)) for accurate signature.
     |  
     |  solve(self) -> 'Solution'
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)
    
    class RecGrid(builtins.object)
     |  RecGrid(nlevel: int, xlower: float, xupper: float, ylower: float, yupper: float, boundaries: 'dict[str, Boundary]')
     |  
     |  Methods defined here:
     |  
     |  ApplyBoundaryMatrix(self, matrix: 'np.array', boundaryMatrix: 'list[list[Boundary]]') -> 'np.array'
     |      通过边界矩阵给当前解施加边界条件
     |  
     |  __init__(self, nlevel: int, xlower: float, xupper: float, ylower: float, yupper: float, boundaries: 'dict[str, Boundary]')
     |      构造网格, nlevel代表当前网格的粗细程度
     |  
     |  constructBoundaryMatrix(self) -> 'list[list[Boundary]]'
     |      由矩形边界给出一个当前level的边界矩阵
     |  
     |  constructGrid(self, nlevel) -> 'RecGrid'
     |      构造网格, nlevel代表当前网格的粗细程度
     |  
     |  constructMatrixFromFunction(self, f0: 'function') -> 'list[list[float]]'
     |      由函数f0得到当前grid上生成矩阵
     |  
     |  constructZeroMatrix(self) -> 'np.array'
     |      由函数f0得到当前grid上生成矩阵
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)
    
    class RelaxationMethod(builtins.object)
     |  Methods defined here:
     |  
     |  GaussSeidel(self, f: 'np.array', boundary: 'tuple[list[list[Boundary]], float]', f0: 'np.array', ds: float = 0.1)
     |      迭代求解一步
     |      f : 当前的解
     |      f0 : 泊松方程的矩阵
     |      grid : 所构造的网格
     |  
     |  Jacobi(self, f: 'np.array', boundary: 'tuple[list[list[Boundary]], float]', f0: 'np.array', ds: float = 0.1)
     |      迭代求解一步
     |      f : 当前的解
     |      f0 : 泊松方程的矩阵
     |      grid : 所构造的网格
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)
    
    class Solution(builtins.object)
     |  Solution(anything)
     |  
     |  Methods defined here:
     |  
     |  __init__(self, anything)
     |      Initialize self.  See help(type(self)) for accurate signature.
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)

FILE
    c:\users\steven_rong\desktop\zju\大三\上\计算物理\lec&proj\project\2dequation.py


