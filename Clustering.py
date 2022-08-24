#                       TASK 1
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from numpy.linalg import eig
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.preprocessing import StandardScaler
df = pd.read_csv("C:/Users/manju/OneDrive/Data Analysis For Business Intelligence/Modules/Semester_1/Data Mining and Neural Networks/Computational_task/Computational Tak_1/wdbc - Copy.csv")

#                       NORMALISE THE DATA
dummy = pd.get_dummies(df['Diagnosis'])
df1 = pd.concat((df,dummy),axis=1)
df1 = df1.drop(['Diagnosis','M'],axis=1)
df1.rename(columns={'B':'Diagnosis'}, inplace = True)
print(df1)
df2 = df1.drop(['ID Number', 'Diagnosis'],axis=1)
print(df2)
x = df1['Diagnosis']
t = df1['Diagnosis'].to_numpy()
print(t)
column = df2.columns
scaler = StandardScaler()
scaler.fit(df2)
standardized_data = scaler.transform(df2)
standardized__data = pd.DataFrame(standardized_data, columns=column)

#                   Principal Component Analysis
pca = PCA(n_components=3)
principalComponents_data = pca.fit_transform(standardized__data)
principalComponents__Data= pd.DataFrame(data = principalComponents_data,columns = ['principal component 1', 'principal component 2','principal component 3'])
print(principalComponents__Data)


#                   Eigen values of correlation matrix
corrMatrix = standardized__data.corr()
#print("              corrMatrix")
print("\t\tcorrMatrix")
print(corrMatrix)
values , vectors = eig(corrMatrix)
print("               Eigen values")
print(values)
#print(vectors)
plt.plot(values)
plt.title('Eigen values')
plt.xlabel('Eigen values')
plt.ylabel('Count')
plt.show()

#                                       TASK 2
print(principalComponents__Data)
principal_component = pd.concat((principalComponents__Data,x),axis=1)
print(principal_component)
sns.histplot(x ='principal component 1', hue ='Diagnosis', data = principal_component)
median1 = 0
plt.axvline(median1,color='black',label='Median')
plt.title("Principal Component 1(PC1)")
plt.xlabel("PC1")
plt.ylabel("Count")
plt.show()
sns.histplot(x ='principal component 2', hue ='Diagnosis', data = principal_component)
median1 = 0.0547195
plt.axvline(median1,color='black',label='Median')
plt.title("Principal Component 2(PC2)")
plt.xlabel("PC2")
plt.ylabel("Count")
plt.show()
sns.histplot(x ='principal component 3', hue ='Diagnosis', data = principal_component)
median1 = -0.072897
plt.axvline(median1,color='black',label='Median')
plt.title("Principal Component 3(PC3)")
plt.xlabel("PC3")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(10,10))
plt.xlabel('Principal Component - 2',fontsize=20)
plt.ylabel('Principal Component - 1',fontsize=20)
plt.title("Principal Component Analysis(PC1 versus PC2)",fontsize=20)
targets = [1, 0]
colors = ['r', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = df1['Diagnosis'] == target
    plt.scatter(principalComponents__Data.loc[indicesToKeep, 'principal component 2']
               , principalComponents__Data.loc[indicesToKeep, 'principal component 1'], c = color, s = 50)
plt.legend(targets,prop={'size': 15})
plt.show()

plt.figure(figsize=(10,10))
plt.xlabel('Principal Component - 3',fontsize=20)
plt.ylabel('Principal Component - 2',fontsize=20)
plt.title("Principal Component Analysis(PC2 versus PC3)",fontsize=20)
targets = [1, 0]
colors = ['r', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = df1['Diagnosis'] == target
    plt.scatter(principalComponents__Data.loc[indicesToKeep, 'principal component 3']
               , principalComponents__Data.loc[indicesToKeep, 'principal component 2'], c = color, s = 50)
plt.legend(targets,prop={'size': 15})
plt.show()

plt.figure(figsize=(10,10))
plt.xlabel('Principal Component - 3',fontsize=20)
plt.ylabel('Principal Component - 1',fontsize=20)
plt.title("Principal Component Analysis(PC1 versus PC3)",fontsize=20)
targets = [1, 0]
colors = ['r', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = df1['Diagnosis'] == target
    plt.scatter(principalComponents__Data.loc[indicesToKeep, 'principal component 3']
               , principalComponents__Data.loc[indicesToKeep, 'principal component 1'], c = color, s = 50)
plt.legend(targets,prop={'size': 15})
plt.show()


#                   TASK3 and Task 4
#               K=2
X= principalComponents__Data.drop(['principal component 3'],axis=1)
data = X.to_numpy()
print(data)

kmeans = KMeans(init="random",n_clusters=2).fit(X)
centroids = kmeans.cluster_centers_
print("                 Centroids")
print(centroids)
labels = kmeans.labels_
print("                    labels")
print(labels)
DB_index1 = davies_bouldin_score(X, labels)
print("                     DB Index")
print(DB_index1)
plt.scatter(principalComponents__Data['principal component 1'],principalComponents__Data['principal component 2'],c= kmeans.labels_.astype(float), alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='r', s=50)
plt.title("K_means = 2(PC1 and PC2)")
plt.show()
print("                     Cross Table")
print(pd.crosstab(t,labels))
X= principalComponents__Data.drop(['principal component 3'],axis=1)
data = X.to_numpy()
print(data)


#               K=3

kmeans = KMeans(init="random",n_clusters=3).fit(X)
centroids = kmeans.cluster_centers_
print("                                 centroids")
print(centroids)
labels = kmeans.labels_
DB_index2 = davies_bouldin_score(X, labels)
print("                                    DB_index2")
print(DB_index2)
plt.scatter(principalComponents__Data['principal component 1'],principalComponents__Data['principal component 2'],c= kmeans.labels_.astype(float), alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='r', s=50)
plt.title("K_means = 3(PC1 and PC2)")
plt.show()
print("                                 Cross table")
print(pd.crosstab(t,labels))
#                   K=5
kmeans = KMeans(init="random",n_clusters=5).fit(X)
centroids = kmeans.cluster_centers_
print("                                     centroids")
print(centroids)
labels = kmeans.labels_
DB_index3 = davies_bouldin_score(X, labels)
print("                                         DB_index3")
print(DB_index3)
plt.scatter(principalComponents__Data['principal component 1'],principalComponents__Data['principal component 2'], c= kmeans.labels_.astype(float), alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='r', s=50)
plt.title("K_means = 5(PC1 and PC2)")
plt.show()
print("                                         Cross table")
print(pd.crosstab(t,labels))


