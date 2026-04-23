import numpy as np
from collections import Counter
import matplotlib.pyplot as plt


X = np.array([[2,3,4],[7,3,5],[3,4,4],[7,8,4],[3,3,7],[5,2,8],[9,1,6],[3,9,5]])
y = np.array(["red","red","red","red","blue","blue","blue","blue"])        
new_point = [6,4,7]

def euclidean_distance(p,q):
    return np.sqrt(np.sum((p-q)**2))
class KNN:
    def __init__(self,radius):
        self.radius = radius
        self.k = 3

    def fit(self,X,y):
        self.X = X
        self.y = y
    def predict(self,new_point):
        distances = []
        near_points = []
        for i in range(len(self.X)):
            distance = euclidean_distance(self.X[i],new_point)
            distances.append([distance,self.y[i]])
            if(distance <= self.radius):
                near_points.append([distance,self.y[i]])
        if(len(near_points) !=0):
            labels = [label for _,label in near_points]
            result = Counter(labels).most_common(1)[0][0]
            return result
            
        distances.sort()
        nearest_neighbors = distances[:self.k]
        
        labels = [label for _,label in nearest_neighbors]
        result = Counter(labels).most_common(1)[0][0]
        return result


clf = KNN(5.10)
clf.fit(X,y)
predicted_color = clf.predict(new_point)
print(predicted_color)

       
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.grid(True)
ax.set_xlim(0,10)
ax.set_ylim(0,10) 
ax.set_zlim(0,10)
ax.scatter(new_point[0],new_point[1],new_point[2],marker='*',color=predicted_color)
i = 0
for point in X:
    ax.scatter(point[0],point[1],point[2],color = y[i])
    i = i+1
for point in X:
    dist = euclidean_distance(point,new_point)
    ax.plot(
    [new_point[0],point[0]],
    [new_point[1],point[1]],
    [new_point[2],point[2]],
    color='black',alpha=0.5,
    linewidth=1,linestyle=':')
    mid = (new_point+point)/2
    ax.text(mid[0],mid[1],mid[2],f"{dist:.2f}")
plt.show()