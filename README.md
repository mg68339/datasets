<h1 align="center">Hi 👋, I'm Mohit Gupta</h1>
<h3 align="center">A passionate frontend developer from India</h3>

- 🔭 I’m currently working on [Event management website](https://github.com/mg68339/Event-Management-System)

- 🌱 I’m currently learning **MERN STACK**

- 📫 How to reach me **mg68339@gmail.com**

<h3 align="left">Connect with me:</h3>
<p align="left">
<a href="https://twitter.com/moohhhiiittt" target="blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/twitter.svg" alt="moohhhiiittt" height="30" width="40" /></a>
<a href="https://fb.com/mg68339" target="blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/facebook.svg" alt="mg68339" height="30" width="40" /></a>
<a href="https://instagram.com/moohhhiiittt" target="blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/instagram.svg" alt="moohhhiiittt" height="30" width="40" /></a>
<a href="https://discord.gg/mg68339" target="blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/discord.svg" alt="mg68339" height="30" width="40" /></a>
</p>

<h3 align="left">Languages and Tools:</h3>
<p align="left"> <a href="https://getbootstrap.com" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/bootstrap/bootstrap-plain-wordmark.svg" alt="bootstrap" width="40" height="40"/> </a> <a href="https://www.cprogramming.com/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/c/c-original.svg" alt="c" width="40" height="40"/> </a> <a href="https://www.w3schools.com/cpp/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/cplusplus/cplusplus-original.svg" alt="cplusplus" width="40" height="40"/> </a> <a href="https://www.w3schools.com/css/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/css3/css3-original-wordmark.svg" alt="css3" width="40" height="40"/> </a> <a href="https://www.w3.org/html/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/html5/html5-original-wordmark.svg" alt="html5" width="40" height="40"/> </a> <a href="https://www.java.com" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/java/java-original.svg" alt="java" width="40" height="40"/> </a> <a href="https://developer.mozilla.org/en-US/docs/Web/JavaScript" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/javascript/javascript-original.svg" alt="javascript" width="40" height="40"/> </a> <a href="https://www.mongodb.com/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/mongodb/mongodb-original-wordmark.svg" alt="mongodb" width="40" height="40"/> </a> <a href="https://www.mysql.com/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/mysql/mysql-original-wordmark.svg" alt="mysql" width="40" height="40"/> </a> <a href="https://nodejs.org" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/nodejs/nodejs-original-wordmark.svg" alt="nodejs" width="40" height="40"/> </a> <a href="https://postman.com" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/getpostman/getpostman-icon.svg" alt="postman" width="40" height="40"/> </a> <a href="https://www.python.org" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="40" height="40"/> </a> <a href="https://reactjs.org/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/react/react-original-wordmark.svg" alt="react" width="40" height="40"/> </a> </p>



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