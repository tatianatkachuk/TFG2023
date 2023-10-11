import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU, SimpleRNN
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer 
from constants import ScalerFunctions, AirportsNames
import obtainMongoData
import threading
import time
from queue import Queue

pd.options.mode.chained_assignment = None

SC = ScalerFunctions()

# DEEP LEARNING PARAMS
LOSS = 'mse'
OPT = 'adam'
METR = ['mse']
EP = 25
VAL_SPL = 0.20
BATCH_S = 300

# MODEL PARAMS
LSTMVAL = [200, 200, 200, 200]
DENSE1 = [10, 10, 10, 10]
DENSE2 = [10, 10, 10, 10]
DENSE3 = [5, 5, 5, 5]
DENSE4 = [3, 3, 3, 3]

# PREDICTION PARAMS
HOURS = 24
PRED = 2
PREDFORHOURS = 4 # lo que se representa en el mapa


def preProcessingData(df, selected_to_analyze, fulldf, alt, airport):
    df['timestamp'] = pd.to_datetime(df['fint']).apply(lambda x: x.timestamp())

    df = df[selected_to_analyze]
    df = df[df['alt'] == alt]

    # TARGET 24 hours
    df['target_dv'] = df.shift(-24)['dv']
    df['target_prec'] = df.shift(-24)['prec']
    df['target_vis'] = df.shift(-24)['vis']

    df = df[fulldf]

    try:
        df = df.replace('NaN', np.nan)
        imputer = KNNImputer(n_neighbors=24, weights='uniform')
        df = df[fulldf]
        
        df_imputed = imputer.fit_transform(df)

        df_processed = pd.DataFrame(df_imputed, columns=df.columns)
        
    except Exception as e:
        print(str(e))

    return df_processed


def processingData(df, selected_to_analyze, selected_to_predict):

    x_full = df[selected_to_analyze]
    y_full = df[selected_to_predict]
    x_full = SC.normalizeMatrix(x_full)
    y_full = SC.normalizeMatrix(y_full)

    

    x_train, x_test, y_train, y_test = train_test_split(
        x_full, y_full, test_size=0.25)

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    return x_full, y_full, x_train, x_test, y_train, y_test


def defineModel(tipo, lstm, dense1, dense2, dense3, dense4):
    model = Sequential()

    if tipo == 'simple':
        model.add(SimpleRNN(units=200, input_shape = (10,1)))
        model.add(Dense(units=5, activation='softmax'))
        model.add(Dense(units=3, activation='linear'))

    if tipo == 'lstm':
        model.add(LSTM(lstm))
        model.add(Dense(units=dense1, activation='relu', input_shape=(10,)))
        model.add(Dense(units=dense2, activation='softmax'))
        model.add(Dense(units=dense3, activation='relu'))
        model.add(Dense(units=dense4, activation='linear'))

    if tipo == 'gru':
        model.add(GRU(units=200,      
                input_shape = (10,1),               
                activation='tanh',             
                recurrent_activation='relu',             
                kernel_initializer='glorot_uniform', 
                recurrent_initializer='orthogonal',   
                bias_initializer='zeros',         
                dropout=0.0,                    
                recurrent_dropout=0.0,       
                return_sequences=False        
                ))
        model.add(Dense(units=5, activation='softmax'))
        model.add(Dense(units=3, activation='linear'))

    return model


def analizePlot(hist, ep, params, col):
    loss_values = hist.history[params[0]]
    val_loss_values = hist.history[params[1]]
    epochs = range(1, ep+1)

    plt.plot(epochs, loss_values, col, label='Training ' + params[2])
    plt.plot(epochs, val_loss_values, 'r', label='Validation ' + params[0])
    plt.title(params[2] + ': Training and validation ' + params[0])
    plt.xlabel('Epochs')
    plt.ylabel(params[0].capitalize())
    plt.legend()
    plt.show()


def makePrediction(df, hours, model):
    # filter = x_full[df['alt'] == altitude]
    df_last_h = df[-hours:]
    result = model.predict(df_last_h)

    return result


def getAltitudes(raw_data, airports):
    # altitudes = [4, 71, 370, 560.12]
    altitudes = []
    # Obtener la altitud con nombre de aeropuerto 
    for a in airports:
        aname = AirportsNames.returnName(a) 
        k = raw_data[raw_data['ubi'] == aname]
        altitudes.append(float(k.iloc[0]['alt']))

    return altitudes


def main(df_raw, alt, airport, lstmval, d1, d2, d3, d4, col, resultQ, hours):

    # Dataframe entera que se utilizará 
    fulldf = ['alt', 'vv', 'vmax', 'dv', 'pres',
              'hr', 'ta', 'tpr', 'prec', 'vis',
              'target_dv', 'target_prec', 'target_vis']
    # Columnas que se usarán para analizar
    selected_to_analyze = ['alt', 'vv', 'vmax', 'dv', 'pres',
                           'hr', 'ta', 'tpr', 'prec', 'vis']
    # Columnas que se usarán para predecir
    selected_to_predict = ['target_dv', 'target_prec', 'target_vis']

    # Parámetros que se quieren analizar

    # Curva de pérdidas
    # lossPlot = True
    lossPlot = False

    # Curva de errores
    errorPlot = True
    # errorPlot = False

    # Validación del modelo
    # Evaluación de la precisión y pérdidas de validación
    # validation = True
    validation = False

    # Matriz de correlación
    # heatmap = True
    heatmap = False

    # Preprocesado de datos
    df_processed = preProcessingData(
        df_raw, selected_to_analyze, fulldf, alt, airport)
    
    # Procesado de datos
    x_full, y_full, x_train, x_test, y_train, y_test = processingData(
        df_processed, selected_to_analyze, selected_to_predict)

    # DEEP LEARNING
    # Elección, definición y compilación del modelo
    # tipo = 'simple'
    tipo = 'lstm'
    # tipo = 'gru'
    model = defineModel(tipo, lstmval, d1, d2, d3, d4)
    model.compile(loss=LOSS, optimizer=OPT, metrics=METR)

    hist = model.fit(x_train, y_train, validation_split=VAL_SPL,
                     epochs=EP, batch_size=BATCH_S, verbose=0)

    # HEATMAP correlación de parámetros
    if heatmap: 
        sns.heatmap(df_processed.corr(), annot=True)
        plt.show()
        # print(df_processed.corr())

    # GRÁFICAS de pérdidas y errores
    if lossPlot:
        params = ['loss', 'val_loss', airport]
        analizePlot(hist, EP, params, col)

    if errorPlot:
        params = ['mse', 'val_mse', airport]
        analizePlot(hist, EP, params, col)
    
    # VALIDACIÓN del modelo
    if validation:
        score = model.evaluate(x_test, y_test, verbose=0)
        print(airport)
        print('Loss on the test set:', score[0])
        print('Accuracy on the test set:', score[1])

    # PREDICCIÓN
    prediction = True
    # prediction = False

    results = [] 
    if prediction:
        r = []
        r.append(airport)
        res = makePrediction(x_full, HOURS, model)
        r.append(str(round(SC.originalMatrix(res)[PRED][0], hours)))
        r.append(str(abs(round(SC.originalMatrix(res)[PRED][1], hours))))
        r.append(str(abs(round(SC.originalMatrix(res)[PRED][2], hours))))
        results.append(r)
        resultQ.put(results)


frontend = True
# frontend = False

if frontend: 
    def deepL():

        airports = ['Barcelona', 'Reus', 'Santiago', 'Granada']
        
        # Inicializar cola de resultados
        resultQ = Queue()
        
        # Obtener todos los datos en bruto de la base de datos
        df_raw = obtainMongoData.returnDataframe()
        # Obtener las alturas para separar los datos en los aeropuertos seleccionados
        alt_values = getAltitudes(df_raw, airports)

        threads = []

        for i in range(len(alt_values)):
            print(i)
            thread = threading.Thread(target=main, args=(
                df_raw, alt_values[i], airports[i], LSTMVAL[i], DENSE1[i], DENSE2[i], DENSE3[i], DENSE4[i], 'b', resultQ, PREDFORHOURS,))
            threads.append(thread)


        for t in threads:
            t.start()
            time.sleep(1)
            t.join()
            time.sleep(1)
    
        print("Todas las tareas han sido completadas")
         # Imprimir resultados
        results = []

        while not resultQ.empty():
            r = resultQ.get()
            results.append(r[0]) 


        return results, PREDFORHOURS


else:   

    if __name__ == '__main__':

        # Elección del (de los) aeropuerto(s) a analizar
        # all_airports = True
        all_airports = False

        if all_airports: airports = ['Barcelona', 'Reus', 'Santiago', 'Granada']
        else: # Escribir nommbre o nombres
            airports = ['Granada']

        # Inicializar cola de resultados
        resultQ = Queue()
        
        # Obtener todos los datos en bruto de la base de datos
        df_raw = obtainMongoData.returnDataframe()
        # Obtener las alturas para separar los datos en los aeropuertos seleccionados
        alt_values = getAltitudes(df_raw, airports)

        threads = []

        for i in range(len(alt_values)):
            print(i)
            thread = threading.Thread(target=main, args=(
                df_raw, alt_values[i], airports[i], LSTMVAL[i], DENSE1[i], DENSE2[i], DENSE3[i], DENSE4[i], 'b', resultQ, PREDFORHOURS,))
            threads.append(thread)


        for t in threads:
            t.start()
            time.sleep(1)
            t.join()
            time.sleep(1)
    
        print("Todas las tareas han sido completadas")

        
        # Imprimir resultados
        results = []

        while not resultQ.empty():
            r = resultQ.get()
            # print(r)
            results.append(r[0])

        print(results)
