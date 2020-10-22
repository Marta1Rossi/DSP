import json

drone_file = open('./listadroni.json')
drone_list = json.load(drone_file)
for i in drone_list['DroneList']:
    print(i['DroneType'])