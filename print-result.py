#! /usr/bin/env python3
# coding: utf-8
import json
import os
import hashlib

class Result:
    
    def __init__(self, **agent_attributes):
        for attr_name, attr_value in agent_attributes.items():
            setattr(self, attr_name, str(attr_value))



def main():
    map_result={}
    
    for file in os.listdir('./resultat_1v1/'):
        for result_attributes in json.load(open('./resultat_1v1/'+file)):
            result = Result(**result_attributes)
            key_result = result.class1+';'+result.class2+';'+result.nb_training_data+';'+result.error+';'+result.iteration_model
            map_result[hashlib.md5(key_result.encode('utf-8')).hexdigest()]=result.class1+';'+result.class2+';'+result.training_duration+';'+result.nb_training_data+';'+result.prediction_duration+';'+result.error+';'+result.iteration_model
    
    for valeur in map_result.values():
        print(valeur)

if __name__ == "__main__":
    main()