import matplotlib.pyplot as plt
import sklearn.datasets as skdata
from sklearn.linear_model import LogisticRegression
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics



numeros = skdata.load_digits()
target = numeros['target']
imagenes = numeros['images']
n_imagenes = len(target)

data = imagenes.reshape((n_imagenes, -1)) # para volver a tener los datos como imagen basta hacer data.reshape((n_imagenes, 8, 8))
print(np.shape(data))


scaler = StandardScaler()
x_train, x_test, y_train, y_test = train_test_split(data, target, train_size=0.5)


X_train = scaler.fit_transform(x_train)
X_test = scaler.transform(x_test)

# Turn up tolerance for faster convergence
clf = LogisticRegression(C=50. / len(X_train), penalty='l1', solver='saga', tol=0.1)
clf.fit(X_train, y_train)
coef = clf.coef_.copy()

scale = np.abs(coef).max()/20
plt.figure(figsize=(10, 5))
for i in range(10):
    l1_plot = plt.subplot(2, 5, i + 1)
    l1_plot.imshow(coef[i].reshape(8, 8),vmin=-scale, vmax=scale)
    l1_plot.set_xticks(())
    l1_plot.set_yticks(())
    l1_plot.set_xlabel('Categoría %i' % i)
    l1_plot.set_title(r'$\vec{\beta_1}$')
    
plt.suptitle('Coeficientes de la regresión para cada categoría')
plt.savefig('coeficientes.png')


predicciones = clf.predict(X_test)
matriz = metrics.confusion_matrix(y_test,predicciones)


plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
plt.figure(figsize=(8,8))
plt.imshow(matriz)
for i in range(0,10):
    for j in range(0,10):
        plt.text(i-0.35,j+0.05,' {:.2f}'.format(float(matriz[i,j])/np.sum(y_test==i)),fontsize=10)
        plt.yticks(np.linspace(0,9,10),np.linspace(0,9,10,dtype=int))
        plt.xticks(np.linspace(0,9,10),np.linspace(0,9,10,dtype=int))
        
plt.title('Matriz de confusión')
plt.savefig('confusion.png')