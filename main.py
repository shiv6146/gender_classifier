import pandas
import graphviz
from sklearn import tree, preprocessing
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Reading dataset
dataset = 'Transformed Data Set - Sheet1.csv'
df = pandas.read_csv(dataset)

# Initialize data preprocessor
label_encoder = preprocessing.LabelEncoder()

# Initializing the classifiers
tree_classifier = tree.DecisionTreeClassifier()
sv_classifier = LinearSVC()
pcpt_classifier = Perceptron()
kn_classifier = KNeighborsClassifier()

# Feature names
features = [df.columns[col] for col in xrange(len(df.columns)-1)]

# Encode feature names and construct rows of features from the dataset
X = [label_encoder.fit_transform(df.iloc[i][:df.iloc[i].size-1].tolist()).tolist() for i in xrange(df.last_valid_index()+1)]
# Target class names
Y = df.Gender.tolist()

# Train your data to match features into targets
tree_classifier.fit(X, Y)
sv_classifier.fit(X, Y)
pcpt_classifier.fit(X, Y)
kn_classifier.fit(X, Y)

# Evaluate accuracy of each classifier by testing it against the same dataset and matching with its target
tree_pred = tree_classifier.predict(X)
tree_score = accuracy_score(Y, tree_pred)
print "Accuracy for Decision tree: " + str(tree_score*100)

sv_pred = sv_classifier.predict(X)
sv_score = accuracy_score(Y, sv_pred)
print "Accuracy for Linear Support Vector classifier: " + str(sv_score*100)

pcpt_pred = pcpt_classifier.predict(X)
pcpt_score = accuracy_score(Y, pcpt_pred)
print "Accuracy for Perceptron Linear Model classifier: " + str(pcpt_score*100)

kn_pred = kn_classifier.predict(X)
kn_score = accuracy_score(Y, kn_pred)
print "Accuracy for KNeighbours classifier: " + str(kn_score*100)

# Visualize the decision tree
dot_data = tree.export_graphviz(tree_classifier, out_file=None, feature_names=features, class_names=['M', 'F'], filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("GenderClassifierTree")

# Select the classifier with highest accuracy score
scores = [tree_score, sv_score, pcpt_score, kn_score]
classifiers = [tree_classifier, sv_classifier, pcpt_classifier, kn_classifier]
best_classifier = classifiers[np.argmax(scores)]

# Get user input data and make prediction
user_input = []

for col in features:
    user_input.append(str(raw_input("What's your " + col + " : ")))

prediction = best_classifier.predict([label_encoder.fit_transform(user_input).tolist()])

print prediction