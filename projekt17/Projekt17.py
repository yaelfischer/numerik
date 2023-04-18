import numpy as np
import os
import numpy.linalg as lin
import matplotlib
import matplotlib.pyplot as plt
from scipy.linalg import solve_triangular
import statistics

script_dir = os.path.dirname(__file__)
rel_path = "Daten/datenPunkte.csv"
abs_file_path1 = os.path.join(script_dir, rel_path)

Q = np.loadtxt(abs_file_path1, delimiter=",")
#Q = np.array([[1,0],[0,1],[-1,0],[0,-1],[0.707,0.707],[-0.707,0.707],[0.707,-0.707],[-0.707,-0.707]])     # circular data -> SW are the same
points = np.copy(Q)
n = len(Q[:,0])

Xm, Ym = np.mean(Q, axis=0)
Q = Q - [Xm, Ym]

# SVD
U, Sigma, V = lin.svd(Q)
#Sigma = np.diag(Sigma)
#Sigma = np.pad(Sigma, [(0, n - len(Sigma)), (0, 0)], mode='constant')

# create SVD fit
line = np.zeros((2,2))
line[0,0] = Xm + (-2.5) * V[0,0]
line[0,1] = Ym + (-2.5) * V[0,1]
line[1,0] = Xm + 1.8 * V[0,0]
line[1,1] = Ym + 1.8 * V[0,1]

plt.scatter(points[:,0], points[:,1])
plt.plot(line[:,0], line[:,1], 'r')

# Ausgleichsgerade
b = points[:,1]
A = np.column_stack((np.ones(n), points[:,0]))
Q1, R1 = lin.qr(A)
alpha = solve_triangular(R1,np.transpose(Q1)@b)

line_x = np.array([-0.29, 3.25])
plt.plot(line_x,alpha[0]+line_x*alpha[1])
plt.quiver([Xm,Xm],[Ym, Ym],V[:,0],V[:,1], scale=1, units='xy', angles='xy', width=0.03)
plt.legend(['Data', 'SVD', 'Ausgleichsgerade', 'Rechts-SV'])
plt.gca().set_aspect('equal')
plt.grid()
plt.savefig('plt1.png')
plt.show()


# Task 4 Facedetection

numberTrainPers = 30
numberTrainPerPers = 9
sizeOfImage = 56 * 68
Q = np.zeros((0,sizeOfImage))


# Training data

for j in range(1,numberTrainPers+1):
    for i in range(1,numberTrainPerPers+1):
        rel_path = ("Daten/Gesichter/s" + str(j) + "/f" + str(i) + ".png")
        abs_file_path2 = os.path.join(script_dir, rel_path)
        image = plt.imread(abs_file_path2)
        faceVec = np.concatenate(image)
        Q = np.append(Q, [faceVec], axis=0)

Q = Q.T

# center  cloud
n, m = np.shape(Q)
meanImage = np.mean(Q, axis=1)
Q = Q - np.tile(meanImage,(Q.shape[1],1)).T

U, _, _ = lin.svd(Q, full_matrices=False)


# Function to Projection and error calculation

def findFace(U, image, dim=50):
    imageVec = np.concatenate(image)

    imageTest = imageVec - meanImage

    # reduce number of used singular values
    Ur = U[:, :dim]

    aprox = meanImage + Ur@(Ur.T@imageTest)
    reconstImage = np.array(np.split(aprox, 68))

    # calculate error
    error = sum(map(sum, np.abs(image-reconstImage)**2))
    print("SSD: " + str(error))

    return reconstImage, error


# test gesicht f10
testDimensions = [5, 15, 30, 100, 150, 200]
rel_path = ("Daten/Gesichter/s37/f1.png")
imagePath = os.path.join(script_dir, rel_path)
origImage = plt.imread(imagePath)
plt.subplot(2, len(testDimensions)+1, 1)
plt.title("Original")
plt.imshow(origImage, cmap='gray')
plt.xticks([])
plt.yticks([])

errVec = []

for i in range(len(testDimensions)):
    image = plt.imread(imagePath)
    reconstImage, error = findFace(U, image, testDimensions[i])
    plt.subplot(2, len(testDimensions)+1, i+2)
    plt.title("Dim: " + str(testDimensions[i]))
    plt.xticks([])
    plt.yticks([])
    plt.imshow(reconstImage, cmap='gray')
    errVec.append(error)

plt.subplot(2, 1, 2)
plt.plot(testDimensions, errVec)
plt.xlabel('Dimension', fontsize=12)
plt.ylabel('Abweichung', fontsize=12)
plt.show()
plt.grid()


# Skyline

faceWidth = 56
skylineWidth = 400

#rel_path = ("Daten/Gesichter/trainedFaces.png")
rel_path = ("Daten/Gesichter/untrainedFaces.png")
#rel_path = ("Daten/Gesichter/ZHAW.png")

imagePath = os.path.join(script_dir, rel_path)
skyline = plt.imread(imagePath)

plt.subplot(3, 1, 1)
#plt.title("FaceDetection")
plt.imshow(skyline, cmap='gray')
plt.yticks([])

dimensions = [1,3,4]
errMean = np.zeros((1,skylineWidth - faceWidth))
errArray = np.zeros((len(dimensions),skylineWidth - faceWidth))
i = 0


# Project skyline and calculate error

plt.subplot(3, 1, 2)
for dimension in dimensions:
    skylineErr = []
    for x in range(0, skylineWidth - faceWidth):
        section = skyline[:,x:x+faceWidth]
        reconstImage, error = findFace(U, section, dimension)
        skylineErr.append(error)
    errMean = np.add(errMean,np.array(skylineErr))
    errArray[i] = np.array(skylineErr)
    i += 1
    plt.plot(range(skylineWidth - faceWidth), skylineErr)


# Plot data

plt.plot(range(skylineWidth - faceWidth), errMean[0,:]/len(dimensions), linestyle='dashed')

leg = ["Dim "+str(x) for x in dimensions]
leg.append("Dim mean")
plt.legend(leg, loc="upper left")
plt.xlim([0, 400])

plt.xlabel('Position im Bild', fontsize=12)
plt.ylabel('Abweichung', fontsize=12)


# Calculate variance

variance = []
for point in errArray.T:
    variance.append(np.var(point))
variance = np.array(variance)


# Detect faces from error data

detect = []
foundFaces = []
for i, point in enumerate(variance):
    if(variance[i] < 90 and errArray[0,i] < 95):
        detect.append(errArray[0,i])
    else:
        if len(detect) != 0:
            foundFaces.append(i - len(detect) + detect.index(min(detect)))
        detect = []

markedImg = skyline.copy()


# Mark faces in image and Display

FaceMarker = np.zeros(len(errArray[0]))
for det in foundFaces:
    FaceMarker[det] = errArray[0,det]
    ## Add boxes to img
    markedImg[:, det-1] = 1
    markedImg[:, det+faceWidth-1] = 1
    markedImg[0, det:det+faceWidth-1] = 1
    markedImg[-1, det:det+faceWidth-1] = 1

#plt.plot(range(skylineWidth - faceWidth), FaceMarker)

plt.subplot(3, 1, 3)
plt.imshow(markedImg, cmap='gray')

plt.show()
plt.grid()



