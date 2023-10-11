# TFG2023
Predicción de condiciones meteorológicas en aeropuertos mediante deep learning

## FRONTEND
Para utilizar el frontend y representar los aeropuertos en el mapa con la información, hay que dejar la variable "frontend" en True (líneas 241-242) 

A continuación, en la terminal ir a la carpeta en la que se encuentra el script manage.py y ejecutar el comando: python .\manage.py runserver

Aparecerá una dirección IP en la terminal. Abrirla. 

NOTA: según el tipo de máquina puede tardar entre 15 segundos y un minuto en ejecutarse todo.

## SIN FRONTEND
Si solamente se desea ver la información de las predicciones sin abrir el mapa, hay que cambiar la variable "frontend" a False. 

En las líneas 165-182 se pueden poner a True los parámetros que se quieran analizar.