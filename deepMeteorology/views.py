import deepLearning_final as deepL
from django.http import JsonResponse
from django.shortcuts import render
from django.http.response import JsonResponse
import json
from pymongo import MongoClient


def markers_list(request):
    results = []

    results, hours = deepL.deepL()

    # while not resultQ.empty():
    #     r = resultQ.get()
    #     results.append(r[0]) 

    # obtiene datos
    connection_string = 'mongodb+srv://tatianatkachuk:VWGeJaHQDfnxx6YS@metdata.d4hd6wn.mongodb.net/'

    client = MongoClient(connection_string)
    col = client['met_data']['dailydata']

    
    markers_details = col.find({})

    different_airports = {}
 
    for marker in markers_details:
        lat = marker['lat']
        long = marker['lon']
        ubi = marker['ubi'].split('/')
        name = ubi[0] 
        idema = marker['idema'] 
        altitude = marker['alt'] 

        if name != 'GIRONA':
        
            new = {
                idema:
                {
                    'name': name,
                    'latitude': lat,
                    'longitude': long,
                    'altitude': altitude
                }
            }
        else: pass

        for key, value in new.items():
            if key not in different_airports:
                different_airports[key] = value
    
    for key, value in different_airports.items():
        
        if value['altitude'] == 4.0: res = results[0]
        elif value['altitude'] == 71.0: res = results[1]
        elif value['altitude'] == 370.0: res = results[2]
        elif value['altitude'] == 560.12: res = results[3]
        
        dv = str(res[1])
        prec = str(res[2])
        vis = str(res[3])
        value['vis'] = vis
        value['prec'] = prec
        value['dv'] = dv 
        
    context = json.dumps({"airports": different_airports, "hours": hours})

    return render(request, 'map.html', json.loads(context))


def predict(request):

    # Lógica para ejecutar el script de Python y obtener los datos
    resultado = "Todo bien"

    # Devolver los datos en formato JSON
    return JsonResponse({'resultado': resultado})
