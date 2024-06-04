import numpy  as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
#Creacion del DF
antiguedad_inmueble = np.array([5, 10, 15, 20, 25,
                                10, 15, 20, 25, 30,
                                15, 20, 25, 30, 35,
                                40, 45, 50, 55, 60,
                                55, 60, 65, 70, 75])
zona_inmubele=np.array([6,6,7,7,8,
               8,9,10,10,10,
               11,11,11,11,12,
               14,16,16,16,16,
               16,17,19,19,22])

precio_inmueble=np.array([11230,9624,13798,3215,19169,
               14982,15419,17286,14232,18092,
               18318,16260,22347,16710,28949,
               27309,32779,29743,30341,34088,
               32435,33909,32263,42067,42295])

datos=np.stack((antiguedad_inmueble,zona_inmubele,precio_inmueble),axis=1)

datos

data_frame_inmueble = pd.DataFrame({"antiguedad": antiguedad_inmueble,
                                    "zona": zona_inmubele,
                                    "precio": precio_inmueble}) 

#Visualizar los datos
data_frame_inmueble.head(1) 
#Grafica
fig=plt.figure(figsize=(4,4))
ax=fig.add_subplot(1,1,1, projection="3d")
ax.scatter(
          data_frame_inmueble["antiguedad"].values,
          data_frame_inmueble["zona"].values,
          data_frame_inmueble["precio"].values,
          marker="o",c="purple",s=500,alpha=0.25)

ax.scatter(
          data_frame_inmueble["antiguedad"].values,
          data_frame_inmueble["zona"].values,
          data_frame_inmueble["precio"].values,
          marker=".",c="black",s=300)

ax.set_xlabel("Antiguedad" ,fontsize=9,color="blue")
ax.set_ylabel("Zona Inmueble",color="blue")
ax.set_zlabel("Precio ($)",color="blue")
plt.tight_layout() 
plt.show()

#Escalamiento de los Datos  
escalador=MinMaxScaler()
escalados=escalador.fit_transform(data_frame_inmueble.values)
escalados=pd.DataFrame(escalados,columns=data_frame_inmueble.columns)
fig=plt.figure(figsize=(12,12))
ax=fig.add_subplot(1,2,1, projection="3d")
ax.scatter(
          escalados["antiguedad"].values,
          escalados["zona"].values,
          escalados["precio"].values,
  
           marker="o",c="purple",s=600,alpha=0.25)

ax.set_title("Datos escalados")
ax.set_xlabel("Antiguedad" ,fontsize=9,color="blue")
ax.set_ylabel("Zona Inmueble",color="blue")
ax.set_zlabel("Precio ($)",color="blue")

ax=fig.add_subplot(1,2,2, projection="3d")
ax.scatter(datos.T[0],datos.T[1],datos.T[2],
           marker="o",c="Blue",s=600,alpha=0.25)
ax.set_title("Datos Originales")
ax.set_xlabel("Antiguedad" ,fontsize=9,color="blue")
ax.set_ylabel("Zona Inmueble",color="blue")
ax.set_zlabel("Precio ($)",color="blue")



plt.show()

escalados.head(2)  

faltantes=escalados.values.copy()
faltantes[[2,7,12,17,22],2]=np.nan
#Obvservamos los faltantes
print(faltantes)

#Imputer
from sklearn.impute import KNNImputer

# Knn uniform

imputer_u=KNNImputer(n_neighbors=5, weights="uniform")

imputer_uniform=imputer_u.fit_transform(faltantes)

# Knn Distance


imputer_d=KNNImputer(n_neighbors=5, weights="distance")

imputer_distance=imputer_d.fit_transform(faltantes)

# Visualización de los datos
print(np.stack((imputer_uniform,imputer_distance),axis=1)[:2])

#Graficación del resultado
fig=plt.figure(figsize=(8,8))
filtro=~np.isnan(faltantes.T[2])
ax=fig.add_subplot(1,1,1, projection="3d")
#Graficamos los valores Originales
ax.scatter(
            escalados["antiguedad"].values[filtro],
          escalados["zona"].values[filtro],
          escalados["precio"].values[filtro],
           label="Originales",
           marker="o",c="blue",s=600,alpha=0.15)

#Graficamos los valores faltantes
ax.scatter(escalados["antiguedad"].values[~filtro],
          escalados["zona"].values[~filtro],
          escalados["precio"].values[~filtro],
           label="NaN/Ausentes",
           marker="*",c="cyan",s=800,alpha=0.65)
#Graficamos los valores remplazados con uniform
ax.scatter(imputer_uniform.T[0][~filtro],
           imputer_uniform.T[1][~filtro],
           imputer_uniform.T[2][~filtro],
           label="IMPUTE UNIFORM",
           marker="D",c="deeppink",s=600)
#Graficamos los valores remplazados con distance
ax.scatter(imputer_distance.T[0][~filtro],
           imputer_distance.T[1][~filtro],
           imputer_distance.T[2][~filtro],
           label="IMPUTE distance",
           marker="s",c="darkorange",s=600)


ax.set_xlabel("Antiguedad" ,fontsize=9,color="blue")
ax.set_ylabel("Zona Inmueble",color="blue")
ax.set_zlabel("Precio ($)",color="blue")
ax.legend(ncol=4)


plt.show()