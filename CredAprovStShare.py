import numpy as np # linear algebra

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score as f1,confusion_matrix, ConfusionMatrixDisplay
#from sklearn.metrics import confusion_matrix
#import scikitplot as skplt #pip install scikit-plot

import pandas as pd 
#import matplotlib.pyplot as plt
import numpy as np 
#import seaborn as sns
from imblearn.over_sampling import SMOTE

import streamlit as st
import time
from PIL import Image
from imagenes_funcionesEspSt import leer_csv,  categorizar, oversample_data, pca_calculos,split_data
modelo=0
entrenamiento=0

print("Calculado......")
#leer archivo
ruta='C:/dOCUMENTOS/Py/pred_creditoBanco/Train.csv'
#df=leer_csv(ruta)
st.set_option('deprecation.showfileUploaderEncoding', False)
file_train = st.sidebar.file_uploader('Cargar archivo Train.csv para entrenar el modelo', type = ['csv'], encoding=None)

if file_train is None:
	st.text(" ¡¡¡ Carga el archivo csv. para entrenamiento !!! ")
else:
	df=leer_csv(file_train)

image = Image.open('C:/dOCUMENTOS/Py/pred_creditoBanco/img_ecommerce.jpg')
st.image(image,use_column_width=True)



try:
    st.title('My first app')
    st.subheader('Predicción de cuentas que continúan o dejan el crédito _v1.0_ ')
    txt='**Dataset descargado de Kaggle: **' + 'https://www.kaggle.com/sakshigoyal7/credit-card-customers'
    st.markdown(txt)
    st.subheader('Data para entrenamiento del modelo: ')
    st.dataframe(df)

    df=df.drop_duplicates()
    txt='**Numero de cuentas : **'+ str(df.shape[0])
    st.markdown(txt)
    st.write(df['EtiquetaReal'].value_counts())

    #categorizar y oversample
    df=categorizar(df)
    oversampled_df,y,ohe_data=oversample_data(df)

    #PCA para ohe_data
    N_COMPONENTS=4
    pca_model, pc_matrix = pca_calculos(N_COMPONENTS,ohe_data)

    #concatenar oversample data con pca de ohe_data
    oversampled_df_with_pcs = pd.concat([oversampled_df,pd.DataFrame(pc_matrix,columns=['PC-{}'.format(i) for i in range(0,N_COMPONENTS)])],axis=1)

    #features seleciccionadas
    X_features = ['Edad','#TransaccionesUltAnio','#Productos','SaldoGastado','CambioMontoTransaccion', 'ImporteTransaccionesUltAnio','Cambio#Transacciones'] #mejor sin PCA-categorical y con las mejores correlaciones
    X = oversampled_df_with_pcs[X_features]
    y = oversampled_df_with_pcs['Label']

    train_x,test_x,train_y,test_y=split_data(X,y)
    rf_pipe = Pipeline(steps =[ ('scale',StandardScaler()), ("RF",RandomForestClassifier(random_state=42)) ])

    #entrenamiento
    rf_pipe.fit(train_x,train_y)
    modelo=1
    txt= " "
    for x in range(6):
        txt=txt + X_features[x] + ', '
    txt='**Caracterísicas seleccionadas para entrenamiento y prueba : **'+ txt
    st.markdown(txt) 
    
except:
  print("Error con el archivo Train.csv") 


imagen=Image.open('C:\dOCUMENTOS\Py\pred_creditoBanco\img_linea.png')
st.image(imagen)
file2 = st.sidebar.file_uploader('Cargar archivo Test.csv para probar el modelo', type = ['csv'], encoding=None)

if file2 is None:
	st.text(" ¡¡¡ Carga el archivo csv. para predecir !!! ")
else:
	df_test=leer_csv(file2)
    #ruta='C:/dOCUMENTOS/Py/pred_creditoBanco/Test.csv'
    #df_test=leer_csv(ruta)
try:

    st.subheader('Información del archivo a predecir. ')
    st.write(df_test.head())
    df_test.EtiquetaReal = df_test.EtiquetaReal.replace({'Attrited Customer':0,'Existing Customer':1})
    originaldata_prediccionRF = rf_pipe.predict(df_test[X_features])
    df_rf=pd.DataFrame(data=originaldata_prediccionRF, columns=['PrediccionRF'])
    df_rf['PrediccionRF']=df_rf.PrediccionRF.replace({0:'Attrited Customer',1:'Existing Customer'})
            
    df_test=pd.concat( [df_rf['PrediccionRF'],df_test], axis=1)

    txt='**Numero de cuentas: **'+ str(df_test.shape[0])
    st.markdown(txt)
    numclase0=df_test['EtiquetaReal'].value_counts()[0]
    txt='**Cuentas de la clase Attrited Customer: **' + str(numclase0)
    st.markdown(txt)
    numclase1=df_test['EtiquetaReal'].value_counts()[1]
    txt='**Cuentas de la clase Existing Customer: **' + str(numclase1)
    st.markdown(txt)
    st.subheader('Resultados de la predicción: ')
    txt='**F1 Score: **'+ str(round(f1(df_test['EtiquetaReal'],originaldata_prediccionRF),2)*100) + " %"
    st.markdown(txt)
    txt='**Matriz de Confusion: ** \n\n Eje Y : True Labels  \nEje X : Predicted Labels'
    st.markdown(txt)
    cm=confusion_matrix(df_test['EtiquetaReal'],originaldata_prediccionRF)
    st.write(cm)
    st.write(pd.DataFrame([{'VP': cm[1,1], 'VN': cm[0,0], 'FP': cm[0,1],  'FN': cm[1,0]}],index=['Cuentas']))
    txt='**Erroes para Attrited Customer: **'+ str(round((cm[0,1]/numclase0)*100,2)) + " %"
    st.markdown(txt)
    txt='**Errores para Existing Customer: **'+ str(round((cm[1,0]/numclase1)*100,2)) + " %"
    st.markdown(txt)
            
    txt='**Archivo resultado: **' + 'La columna PredicionRF contiene las predicciones para cada cuenta sobre si será un Existing Customer o Attrited Customer'
    st.markdown(txt)    
    df_test.EtiquetaReal = df_test.EtiquetaReal.replace({0:'Attrited Customer',1:'Existing Customer'})
    st.write(df_test)    
    rutaGuardarPredicciones=st.text_input('Indique la ruta para guardar el archivo Predicciones.csv con los resultados:')
    if rutaGuardarPredicciones:
        txt=rutaGuardarPredicciones +'\Predicciones.csv'
        df_test.to_csv(txt,index=False)

        st.markdown('**Resultados con predicciones guardado en **' + txt)
except:
  print("Error con el archivo Test.csv")


imagen=Image.open('C:\dOCUMENTOS\Py\pred_creditoBanco\img_linea.png')
st.image(imagen)
st.subheader('Introduzca valores para una cuenta: ')

def get_info_user():
    edad=st.text_input('Edad:')
    transacciones=st.text_input('#Transacciones:')
    productos=st.text_input('#Productos:')
    saldogastado=st.text_input('Saldo Gastado:')
    cambiomontotransaccion=st.text_input('Cambio Monto Transaccion:')
    importetransacciones=st.text_input('Importe Transacciones:')
    cambionumtransacciones=st.text_input('Cambio #Transacciones:')

    user_data={'Edad':edad,
            '#Transacciones':transacciones, 
            '#Productos':productos, 
            'Saldo Gastado':saldogastado, 
            'Cambio Monto Transaccion':cambiomontotransaccion,
            'Importe Transacciones': importetransacciones,
            'Cambio #Transacciones': cambionumtransacciones
            }

    features=pd.DataFrame(user_data,index=[0])
    
    for x in range(features.shape[1]):
        if features[ features.columns[x]].values:
            pass
        else:
            features[ features.columns[x]]=0
    return features

user_input=get_info_user()

if user_input.all:   
    left_column, right_column = st.beta_columns(2)
    btn_predecir = left_column.button('Predecir resultado para la cuenta')
    if btn_predecir:
        if modelo==1:
            label_prediccion=rf_pipe.predict(user_input)
            st.subheader('Resultado: ')
            if label_prediccion==1:
                st.subheader('Existing Customer')
            elif label_prediccion==0:
                st.subheader('Attrited Customer')
        else:
            st.subheader('No existe modelo de entrenamiento')

print("\n Listo, favor de revisar el archivo Predicciones")