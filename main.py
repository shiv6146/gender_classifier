import pandas
from sklearn import tree, preprocessing

dataset = 'Transformed Data Set - Sheet1.csv'

classifier = tree.DecisionTreeClassifier()

label_encoder = preprocessing.LabelEncoder()

df = pandas.read_csv(dataset)

X = [label_encoder.fit_transform(df.iloc[i][:df.iloc[i].size-1].tolist()).tolist() for i in xrange(df.last_valid_index()+1)]
Y = df.Gender.tolist()

classifier = classifier.fit(X, Y)

user_input = []

for col in xrange(len(df.columns)-1):
    user_input.append(str(raw_input("What's your " + df.columns[col] + " : ")))

prediction = classifier.predict([label_encoder.fit_transform(user_input).tolist()])

print prediction