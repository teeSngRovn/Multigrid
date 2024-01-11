import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.collections import LineCollection

class PoissonSolverWithNeumann():
    def __init__(self, funcMatrix, delta, times=1, omega = 1):
        self.func = funcMatrix
        self.RVec = np.array(funcMatrix).flatten()
        self.omega = omega
        self.nx = len(funcMatrix)
        self.ny = len(funcMatrix[0])
        self.delta = delta
        self.times = times
        self.Solution = self.GetSolution(funcMatrix,times)

    def GetSolution(self, func, time):
        solution = [0] * self.nx*self.ny 
        for i in range(time):
            sol = self.Iteration(solution)
            print(np.linalg.norm(np.array(sol) - solution))
            # print(sol[104])
            # test = np.array(sol)
            # print(np.where(test<-10**(-9)))
            solution = sol

        solution = np.array(solution).reshape(self.nx,self.ny)
        return solution

    def GetDiff(self, mat):
        pass

    def Iteration(self, sol):
        result = list()
        for i in range(len(sol)):
            #print(i)
            temp = self.GetFuncValueAtRowI(i)
            RowAndPos = self.GetMatrixRowAndNonZeroPos(i)
            row = RowAndPos[0]
            pos = RowAndPos[1]
            #print("temp = " + str(row[i])+"i = "+str(i))
            test = list()
            for j in pos:
                if j < i:
                    temp -= row[j] * result[j]
                elif j >= i:
                    temp -= row[j] * sol[j]
                test.append(row[j])
            # print(row[i] > 0)
            if (np.sum(np.array(test)) < 0): 
                print(test,pos,i,np.sum(row))
                # print(self.GetRowAndColAtPosI(104))
            temp /= row[i]
            temp *= self.omega
            temp += sol[i]
            # print(temp)
            result.append(temp)
        return result

    def GetMatrixRowAndNonZeroPos(self, i):
        result = self.GetRowAndColAtPosI(i)
        # print(result)
        RowAndPos = self.GetRowAtPosIJ(result[0], result[1])
        return RowAndPos

    def GetFuncValueAtRowI(self, i):
        result = self.GetRowAndColAtPosI(i)
        #print(result)
        value = self.GetFuncAtPosIJ(result[0], result[1])
        #print(value)
        return value

    def GetRowAndColAtPosI(self, i):
        row = i // self.ny
        col = i % self.ny
        return [row, col]

    def GetFuncAtPosIJ(self, i, j):
        vec = float()
        delta = self.delta
        if (0<i<self.nx-1 and 0<j<self.ny-1):
            vec = self.func[i][j] * (delta ** 2)
        elif (i==0):
            if (j==0):
               vec = 0.25 * self.func[i][j] * (delta ** 2)
            elif (j==self.ny-1):
               vec = 0.25 * self.func[i][j] * (delta ** 2)
            else:
               vec = 0.5 * self.func[i][j] * (delta ** 2)
        elif (i==self.nx-1):
            if (j==0):
               vec = 0.25 * self.func[i][j] * (delta ** 2)
            elif (j==self.ny-1):
               vec = 0.25 * self.func[i][j] * (delta ** 2)
            else:
               vec = 0.5 * self.func[i][j] * (delta ** 2)
        elif (j==0):
           vec = 0.5 * self.func[i][j] * (delta ** 2)
        elif (j==self.ny-1):
           vec = 0.5 * self.func[i][j] * (delta ** 2)

        return vec

    def GetRowAtPosIJ(self, i, j):
        nx = self.nx
        ny = self.ny
        pos = list()
        pos.append(self.GetPointPosInRowAtPosIJ(i,j))
        row = [0] * self.nx * self.ny
        if (0<i<self.nx-1 and 0<j<self.ny-1):
            row[self.GetPointPosInRowAtPosIJ(i,j)] = 4
            row[self.GetPointPosInRowAtPosIJ(i+1,j)] = -1
            row[self.GetPointPosInRowAtPosIJ(i-1,j)] = -1
            row[self.GetPointPosInRowAtPosIJ(i,j+1)] = -1
            row[self.GetPointPosInRowAtPosIJ(i,j-1)] = -1
            pos.append(self.GetPointPosInRowAtPosIJ(i+1,j))
            pos.append(self.GetPointPosInRowAtPosIJ(i-1,j))
            pos.append(self.GetPointPosInRowAtPosIJ(i,j+1))
            pos.append(self.GetPointPosInRowAtPosIJ(i,j-1))
        elif (i==0):
            if (j==0):
                row[self.GetPointPosInRowAtPosIJ(i,j)] = 1
                row[self.GetPointPosInRowAtPosIJ(i+1,j)] = -0.5
                row[self.GetPointPosInRowAtPosIJ(i,j+1)] = -0.5
                pos.append(self.GetPointPosInRowAtPosIJ(i+1,j))
                pos.append(self.GetPointPosInRowAtPosIJ(i,j+1)) 
            elif (j==self.ny-1):
                row[self.GetPointPosInRowAtPosIJ(i,j)] = 1
                row[self.GetPointPosInRowAtPosIJ(i+1,j)] = -0.5
                row[self.GetPointPosInRowAtPosIJ(i,j-1)] = -0.5
                pos.append(self.GetPointPosInRowAtPosIJ(i+1,j))
                pos.append(self.GetPointPosInRowAtPosIJ(i,j-1))
            else:
                row[self.GetPointPosInRowAtPosIJ(i,j)] = 2
                row[self.GetPointPosInRowAtPosIJ(i+1,j)] = -1
                row[self.GetPointPosInRowAtPosIJ(i,j-1)] = -0.5
                row[self.GetPointPosInRowAtPosIJ(i,j+1)] = -0.5
                pos.append(self.GetPointPosInRowAtPosIJ(i+1,j))
                pos.append(self.GetPointPosInRowAtPosIJ(i,j-1))
                pos.append(self.GetPointPosInRowAtPosIJ(i,j+1))
        elif (i==self.nx-1):
            if (j==0):
                row[self.GetPointPosInRowAtPosIJ(i,j)] = 1
                row[self.GetPointPosInRowAtPosIJ(i-1,j)] = -0.5
                row[self.GetPointPosInRowAtPosIJ(i,j+1)] = -0.5
                pos.append(self.GetPointPosInRowAtPosIJ(i,j+1))
                pos.append(self.GetPointPosInRowAtPosIJ(i-1,j))
            elif (j==self.ny-1):
                row[self.GetPointPosInRowAtPosIJ(i,j)] = 1
                row[self.GetPointPosInRowAtPosIJ(i-1,j)] = -0.5
                row[self.GetPointPosInRowAtPosIJ(i,j-1)] = -0.5
                pos.append(self.GetPointPosInRowAtPosIJ(i-1,j))
                pos.append(self.GetPointPosInRowAtPosIJ(i,j-1))
            else:
                row[self.GetPointPosInRowAtPosIJ(i,j)] = 2
                row[self.GetPointPosInRowAtPosIJ(i-1,j)] = -1
                row[self.GetPointPosInRowAtPosIJ(i,j-1)] = -0.5
                row[self.GetPointPosInRowAtPosIJ(i,j+1)] = -0.5
                pos.append(self.GetPointPosInRowAtPosIJ(i-1,j))
                pos.append(self.GetPointPosInRowAtPosIJ(i,j-1))
                pos.append(self.GetPointPosInRowAtPosIJ(i,j+1))
        elif (j==0):
            row[self.GetPointPosInRowAtPosIJ(i,j)] = 2
            row[self.GetPointPosInRowAtPosIJ(i,j+1)] = -1
            row[self.GetPointPosInRowAtPosIJ(i-1,j)] = -0.5
            row[self.GetPointPosInRowAtPosIJ(i+1,j)] = -0.5
            pos.append(self.GetPointPosInRowAtPosIJ(i,j+1))
            pos.append(self.GetPointPosInRowAtPosIJ(i-1,j))
            pos.append(self.GetPointPosInRowAtPosIJ(i+1,j))
        elif (j==self.ny-1):
            row[self.GetPointPosInRowAtPosIJ(i,j)] = 2
            row[self.GetPointPosInRowAtPosIJ(i,j-1)] = -1
            row[self.GetPointPosInRowAtPosIJ(i+1,j)] = -0.5
            row[self.GetPointPosInRowAtPosIJ(i-1,j)] = -0.5
            pos.append(self.GetPointPosInRowAtPosIJ(i,j-1))
            pos.append(self.GetPointPosInRowAtPosIJ(i+1,j))
            pos.append(self.GetPointPosInRowAtPosIJ(i-1,j))

        #print(row[self.GetPointPosInRowAtPosIJ(i,j)])
        return [row,pos]
            
    def GetPointPosInRowAtPosIJ(self, i, j):
        result = i * self.ny + j
        return result

class GRID():
    def __init__(self, nx, ny, width, height):
        self.nx = nx
        self.ny = ny
        self.cellx = nx - 1
        self.celly = ny - 1
        self.width = width
        self.height = height
        self.deltax = width / self.cellx
        self.deltay = height / self.celly
        self.x = np.array([[self.deltax * i for i in range(ny)] for j in range(nx)])
        self.y = np.array([[self.deltay * j for i in range(ny)] for j in range(nx)])
        self.i = np.array([[i for i in range(ny - 1)] for j in range(nx - 1)])
        self.j = np.array([[j for i in range(ny - 1)] for j in range(nx - 1)])
        self.cx = np.array([[width / (nx - 2) * j for i in range(ny-1)] for j in range(nx-1)])
        self.cy = np.array([[height / (ny - 2) * i for i in range(ny-1)] for j in range(nx-1)])
        self.cell = np.array(self.compute_area())

    def compute_total_cell_area(self):
        result = np.sum(self.cell)
        return result

    def compute_area(self):
        result = list()
        for i in range(self.cellx):
            temp = list()
            for j in range(self.celly):
                area = self.compute_cell_area(i, j)
                temp.append(area)
            result.append(temp)
        return result

    def compute_cell_area(self, i, j):
        # south west point of cell
        pointsw = [self.x[i][j], self.y[i][j]]
        # south east point of cell
        pointse = [self.x[i + 1][j], self.y[i + 1][j]]
        # north west point of cell
        pointnw = [self.x[i][j + 1], self.y[i][j + 1]]
        # north east point of cell
        pointne = [self.x[i + 1][j + 1], self.y[i + 1][j + 1]]

        # compute area
        result = self.QuadrilateralArea(pointsw, pointse, pointne, pointnw)
        return result

    # The sequence of the point should able to formulate a contour without intersection
    def QuadrilateralArea(self, p1, p2, p3, p4):
        result = 0
        result += np.abs(self.TriangleArea(p1, p2, p3)) + np.abs(self.TriangleArea(p1, p3, p4))
        return result

    def TriangleArea(self, p1, p2, p3):
        x1 = p1[0]; y1 = p1[1]
        x2 = p2[0]; y2 = p2[1]
        x3 = p3[0]; y3 = p3[1]
        result = 0.5 * ((x2 - x1)*(y3 - y1) + (y2 - y1)*(x3 - x1))
        return abs(result)

    def compute_loss(self, brightness):
        total_area = self.compute_total_cell_area()
        loss = self.cell / total_area - brightness
        # loss -= np.sum(loss)/(self.cellx*self.celly)
        return loss

    def step_grid(self, gradx, grady,step = 10):
        gradx = np.pad(gradx, ((0, 1),(0, 1)), mode='constant')
        # gradx[0,:] = 0
        # gradx[:,0] = 0
        grady = np.pad(grady, ((0, 1),(0, 1)), mode='constant')
        # grady[0,:] = 0
        # grady[:,0] = 0
        maxX = np.max(gradx)
        maxY = np.max(grady)
        self.x += gradx * self.deltax / maxX
        self.y += grady * self.deltay / maxY
        self.cell = np.array(self.compute_area())
        print(np.where(self.cell<0))

def read_image(path):
    result = cv2.imread(path)
    return result

def output_image(img):
    cv2.imwrite("output.jpg",img)

def convert2gray(img):
    result = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return result

def normalize_gray(img):
    total = cv2.sumElems(img)[0]
    result = img / total
    return [result, total]

def plotMesh(meshx, meshy):
    # z = np.zeros((meshx.shape[0],meshx.shape[1]))
    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # ax.set_xlim(0, 0.15)
    # ax.set_ylim(0, 0.15)
    # surf = ax.plot_surface(meshx,meshy + 0.0032,z)
    segs1 = np.stack((meshx[:,[0,-1]],meshy[:,[0,-1]]), axis=2)
    segs2 = np.stack((meshx[[0,-1],:].T,meshy[[0,-1],:].T), axis=2)
    plt.gca().add_collection(LineCollection(np.concatenate((segs1, segs2))))
    plt.autoscale()
    # plt.scatter(meshx, meshy)
    plt.show()

def plotGrad(func):
    gradx, grady = np.gradient(func)
    plt.quiver(-grid.i,-grid.j, gradx, grady)
    plt.show()

def plot(sol):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # ax.set_zlim(-0.1, 0.1)
    surf = ax.plot_surface(-grid.i,-grid.j, sol, cmap=cm.coolwarm,linewidth=0, antialiased=False)
    plt.show()


# imgae init
img = read_image("test.jpg")
brightness  = convert2gray(img)
result = normalize_gray(brightness)
brightness = result[0]
total_brightness = result[1]
pix_width = int(brightness.shape[0])
pix_height = brightness.shape[1]

# window init
delta = 0.1 / pix_width
w = delta * pix_width   # width
h = delta *  pix_height # height

# grid init
nx = pix_width + 1
ny = pix_height + 1
grid = GRID(nx, ny, w, h)

# compute loss and iterate
eps = 0.000000001
loss = grid.compute_loss(brightness)
# loss = loss / np.max(loss)

# print(np.max(loss))
plot(loss)
plotMesh(grid.x,grid.y)
# plot(grid.cell)
k = 5
while k >= 0:
    print("loss = "+str(np.max(loss)))
    poisson = PoissonSolverWithNeumann(loss, delta,10,1.8)
    sol = poisson.Solution
   # plotGrad(sol)
    gradx, grady = np.gradient(sol)
    grid.step_grid(gradx,grady)
    loss = grid.compute_loss(brightness)
    plotMesh(grid.x,grid.y)
    if (k==0):
        plot(grid.cell)
        plotMesh(grid.x,grid.y)
    k-=1

# while np.max(loss) > eps:
#     plot(loss)
#     poisson = PoissonSolverWithNeumann(loss, delta,10,0.5)
#     sol = poisson.Solution
#     gradx, grady = np.gradient(sol)
#     grid.step_grid(gradx,grady)
#     loss = grid.compute_loss(brightness)
#     print("loss = "+str(np.max(loss)))









#output_image(loss * total_brightness)
# poisson solve
#loss = grid.cellX + grid.cellY - (grid.cellX ** 2 + grid.cellY ** 2)
#plt.show()
#plt.savefig("figure2.jpg")
#plt.savefig("figure1.jpg")

# Make data.
#mean = np.sum(sol)
#print("sum = " + str(mean))

