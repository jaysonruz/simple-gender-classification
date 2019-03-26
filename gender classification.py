from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

print('Greetings plz enter your height in cm:')
x = int(input())
#
print('Greetings plz enter your weight in kg:')
y = int(input())
#
print('Greetings plz enter your shoe_size in Euro Sizes:')
z = int(input())
#


clf = tree.DecisionTreeClassifier()
knn = KNeighborsClassifier()
Naive = GaussianNB()
# CHALLENGE - create 3 more classifiers...

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']


# CHALLENGE - ...and train them on our data
clf = clf.fit(X, Y)
knn = knn.fit(X,Y)
Naive = Naive.fit(X,Y)

prediction1 = clf.predict([[x, y, z]])
prediction2= knn.predict([[x, y, z]])
prediction3 = Naive.predict([[x, y, z]])

# CHALLENGE compare their reusults and print the best one!

print('clf:',prediction1,'knn:',prediction2,'naive_bayes:',prediction3)