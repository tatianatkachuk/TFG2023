from sklearn.preprocessing import MinMaxScaler

class ScalerFunctions:
    def __init__(self):
        self.scaler = MinMaxScaler()
    
    def normalizeMatrix(self, matrix):
        normalizedMatrix = self.scaler.fit_transform(matrix)
        return normalizedMatrix

    def originalMatrix(self, matrix):
        originalMatrix = self.scaler.inverse_transform(matrix)
        return originalMatrix
    
class AirportsNames:
    def returnName(n):
        if n == 'Barcelona': name = 'BARCELONA/AEROPUERTO'
        elif n == 'Reus': name = 'REUS/AEROPUERTO'
        elif n == 'Santiago': name = 'SANTIAGO DE COMPOSTELA/LABACOLLA'
        elif n == 'Granada': name = 'GRANADA/AEROPUERTO'
        
        return name