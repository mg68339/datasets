import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori

store_data = pd.read_csv('/content/store_data.csv', header=None) 
num_records = len(store_data)
store_data.head()  

records=[]
for i in range(0,num_records):
  records.append([str(store_data.values[i,j]) for j in range(0,20)])

association_rules = apriori(records, min_support=0.0045, min_confidence=0.2, min_lift=3, min_length=2)
association_results = list(association_rules)

print(len(association_results))
print(association_results[0])  


for item in association_results:                                        
    pair = item[0] 
    items = [x for x in pair]
    print("items: " + items[0] + " + " + items[1])
    print("Support: " + str(item[1]))
    print("Confidence: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))
    print("=====================================")

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import tree

df= pd.read_csv('/content/salaries.csv')
df.head()


inputs = df.drop('salary_more_then_100k',axis='columns')
target= df['salary_more_then_100k']

le_company = LabelEncoder()
le_job = LabelEncoder()
le_degree = LabelEncoder()

inputs['company_n']=le_company.fit_transform(inputs['company'])
inputs['job_n']=le_company.fit_transform(inputs['job'])
inputs['degree_n']=le_company.fit_transform(inputs['degree'])

inputs


inputs_n = inputs.drop(['company','job','degree'],axis='columns')

inputs_n

model = tree.DecisionTreeClassifier()

model.fit(inputs_n,target)

model.score(inputs_n,target)

model.predict([[2,1,0]])

# datasets

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

df = pd.read_csv('/content/student_clustering.csv')
df.head()

X=df.iloc[:,:].values
km= KMeans(n_clusters=4)
y_means=km.fit_predict(X)

plt.scatter(X[y_means == 0,0], X[y_means == 0,1],color ='red')
plt.scatter(X[y_means == 1,0], X[y_means == 1,1],color ='green')
plt.scatter(X[y_means == 2,0], X[y_means == 2,1],color ='blue')
plt.scatter(X[y_means == 3,0], X[y_means == 3,1],color ='yellow')