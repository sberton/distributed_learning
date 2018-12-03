#! /usr/bin/env python3
# coding: utf-8

import argparse
import re
import os
import logging as lg
import json
import matplotlib.pyplot as plt
from random import randint
from pyspark.sql import Row
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.mllib.classification import SVMWithSGD
from pyspark.mllib.regression import LabeledPoint
from pyspark.ml.feature import StringIndexer

"""
Use with 
PYSPARK_PYTHON=python3 ../test/code/spark-2.3.1-bin-hadoop2.7/bin/spark-submit ./pet-classification-spark.py -d features -c1 keeshond -i 100
"""
sc = SparkContext()
spark = SparkSession.builder.getOrCreate()


def parse_arguments():
    """ Retrieving arguments
    Parsing argument with argparse module
    """
    parser = argparse.ArgumentParser(description='Process pet classification...', 
                                     prog='pet-classification-spark.py')
    parser.add_argument("-d", "--directorie", help="""datasets directorie""", 
                        required=True)
    parser.add_argument("-v", "--verbose", action='store_true', 
                        help="""Make the application talk!""", required=False)
    parser.add_argument("-n", "--nb_training_data", default=100, nargs='+',  
                        help="""number of training data""", type=int, required=False)
    parser.add_argument("-i", "--iteration_model", 
                        help="""Number of iteration for SVMWithSGD training model""", 
                        required=False,
                        nargs='*',
                        default=100,
                        type=int
                        )
    parser.add_argument("-c1", "--class1", 
                        help="""Abyssinian, Bulldog...""",
                        required=True)
    parser.add_argument("-c2", "--class2", 
                        help="""Abyssinian, Bulldog... All if missing""",
                        required=False,
                        default='All')
    return parser.parse_args()

def main():
    args = parse_arguments()
    result = []
    if args.verbose:
        lg.basicConfig(level=lg.INFO)
    try:
        directorie = args.directorie
        nb_training_data_list = args.nb_training_data
        iteration_model_list = args.iteration_model
        class1 = args.class1
        class2 = args.class2
        if not os.path.exists(directorie):
            raise FileNotFoundError('directorie {} does not exist'.format(directorie))
    except FileNotFoundError as no_directorie:
        lg.critical(no_directorie)
    else:  
        lg.info('#################### Starting pet-classification ######################')
        lg.info('Class 1 is %s', class1)
        lg.info('Class 2 is %s', class2)
        step = 0
        
        for nb_training_data in nb_training_data_list:
            for iteration_model in iteration_model_list:
                lg.info('Number of training datas is %s', nb_training_data)
                lg.info('Number of iterations model is %s', iteration_model)
                train_data_rdd = sc.textFile(directorie+'/*.json')\
                                   .filter(lambda line: line.split(', ')[0] in (class1, class2) or class2 == 'All')\
                                   .filter(lambda line: int(line.split(',')[1]) <= nb_training_data)\
                                   .map(lambda line: Row(label=0.0, features=line.split(', ')[2:]) 
                                        if line.split(', ')[0] == class1 
                                        else Row(label=1.0, features=line.split(', ')[2:]))\
                                   .map(lambda line: LabeledPoint(line.label, line.features))

                lg.info('%s features for training datas', train_data_rdd.count())
                test_data_rdd = sc.textFile(directorie+'/*.json')\
                                  .filter(lambda line: line.split(',')[0] in (class1, class2) or class2 == 'All')\
                                  .filter(lambda line: int(line.split(', ')[1]) > nb_training_data)\
                                  .map(lambda line: Row(label=0.0, features=line.split(', ')[2:]) 
                                       if line.split(', ')[0] == class1 
                                       else Row(label=1.0, features=line.split(', ')[2:]))\
                                  .map(lambda row: LabeledPoint(row.label, row.features))
                lg.info('%s features for test datas', test_data_rdd.count())
                model = SVMWithSGD.train(train_data_rdd, iterations=iteration_model)

                predictions = test_data_rdd.map(lambda row: (row.label, float(model.predict(row.features))))

                train_error = predictions.filter(lambda lp: lp[0] != lp[1]).count() \
                                                 / float(predictions.count())
                lg.info('Test Error ================>%s', str(train_error))

                step += 1
                result.append({"step" : step, "class1" : class1, "class2" : class2,
                                   "iteration_model" : iteration_model, 
                                   "nb_training_data" : nb_training_data, "error" :  train_error})

        with open('result.json', 'w') as result_file:
            json.dump(result, result_file)
    finally:
        lg.info('#################### Ending pet-classification ######################')
        input("press ctrl+c to exit")

if __name__ == "__main__":
    main()
