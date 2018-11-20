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
from pyspark.mllib.classification import SVMWithSGD
from pyspark.mllib.regression import LabeledPoint
from pyspark.ml.feature import StringIndexer

sc = SparkContext()
spark = SparkSession.builder.getOrCreate()

""" Make a dictionnary of features by class
Each classe key contains the list of their features
"""
def load_features(directorie):
    classe_feature = []
    features = []
    for file in os.listdir(directorie):
    #for file in ["Abyssinian_118.jpg.json", "Abyssinian_131.jpg.json", "Abyssinian_139.jpg.json"]:
        #Extract class from filename wich is the dictionnarie key
        current_class = re.sub(r'[0-9]', '', file)[:-9].strip('_')
        features = json.load(open(directorie+"/"+file, "r"))
        features.insert(0, current_class.title())
        #update classe list
        classe_feature.append(features)
    return classe_feature

def split_data(classe_features,nb_classes_features):
    classe_feature_training = []
    classe_feature_test = []
    classe_counter = {}
    for classe_feature in classe_features:
        classe = classe_feature[0]
        if classe not in classe_counter:
            classe_counter[classe] = 0
        index = classe_counter[classe]
        if index < nb_classes_features:
            classe_feature_training.append(classe_feature)
        else:
            classe_feature_test.append(classe_feature)
        classe_counter[classe] += 1
    try:
        #empty sequences are false
        if not classe_feature_test: 
            raise Exception('No test data')
    except IndexError as no_test_data:
        lg.error(no_test_data)        
    return classe_feature_training, classe_feature_test

""" Retrieving arguments
Parsing argument with argparse module
"""
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directorie", help="""datasets directorie""")
    parser.add_argument("-v", "--verbose", action='store_true', 
                        help="""Make the application talk!""")
    parser.add_argument("-n", "--nb_features", 
                        help="""Number of training datas""", type=int, choices=range(1, 500))
    return parser.parse_args()

"""
Creating a dataframe with param list
"""
def load_dataframe(classe_feature_list):
    rdd = sc.parallelize(classe_feature_list)\
            .map(lambda feature_list: Row(label=feature_list[0],
                                          feature=feature_list[1:]))
    return spark.createDataFrame(rdd)

def train_model(dataframe):
    return SVMWithSGD.train(dataframe)

def main():
    args = parse_arguments()
    if args.verbose:
        lg.basicConfig(level=lg.DEBUG)
    try:
        directorie = args.directorie
        nb_features = int(args.nb_features)
        if directorie == None:
            raise Warning('You must indicate a datasets directorie!')
    except Warning as no_directorie:
        lg.warning(no_directorie)
    else:
        classe_feature = load_features(directorie)
        #print(len(classe_feature))
        classe_feature_training, classe_feature_test = split_data(classe_feature,nb_features)
        training_df=load_dataframe(classe_feature_training)
        test_df=load_dataframe(classe_feature_test)
        #print(training_df.count())
        label_indexer = StringIndexer(inputCol="label", outputCol="label_index")
        label_indexer_transformer = label_indexer.fit(training_df)
        training_df = label_indexer_transformer.transform(training_df)
        #training_df.show(truncate=True)
        #mymodel=train_model(training_df)

    finally:
        lg.info('#################### Analysis is over ######################')
    
    #input("press ctrl+c to exit")

if __name__ == "__main__":
    main()
