import pybaobabdt
import pandas as pd
from scipy.io import arff
from sklearn.tree import DecisionTreeClassifier

data = arff.loadarff('winequality-red.arff')
df   = pd.DataFrame(data[0])

y = list(df['class'])
features = list(df.columns)
features.remove('class')
X = df.loc[:, features]

clf = DecisionTreeClassifier().fit(X,y)

ax = pybaobabdt.drawTree(clf, size=10, dpi=72, features=features)

ax.get_figure().savefig('tree.png', format='png', dpi=300, transparent=True)