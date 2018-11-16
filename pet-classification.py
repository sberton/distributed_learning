#! /usr/bin/env python3
# coding: utf-8

import argparse
import sys
import re
import os
import logging as lg
import json
from pyspark.sql import Row
from pyspark import SparkContext
from pyspark.sql import SparkSession


sc = SparkContext()
spark = SparkSession.builder.getOrCreate()

""" Make a dictionnary of feature by class
Each classe key contains the list of their features
"""
def load_features(directorie):
    classe_feature = {}
    features = []
    for file in os.listdir(directorie):
        #Extract class from filename wich is the dictionnarie key
        current_class = re.sub(r'[0-9]', '', file)[:-9].strip('_')
        #retrieve current list if exist
        if current_class in classe_feature:
            features = [classe_feature.pop(current_class)]
        #add current feature to the class features list 
        features.append(json.load(open(directorie+"/"+file)))
        #update dictionnarie key
        classe_feature.update({current_class.title() : features})
    return classe_feature


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directorie", help="""datasets directorie""")
    return parser.parse_args()

def main():
    args = parse_arguments()
    try:
        directorie = args.directorie
        if directorie == None:
            raise Warning('You must indicate a datasets directorie!')
    except Warning as no_directorie:
        lg.warning(no_directorie)
    else:
        classe_feature = load_features(directorie)

    finally:
        lg.info('#################### Analysis is over ######################')
    
    #input("press ctrl+c to exit")

if __name__ == "__main__":
    main()
