#! /usr/bin/env python3
# coding: utf-8

import argparse
import logging as lg
import json
import os.path
import time
from pyspark.sql import Row
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.regression import LabeledPoint

"""
Use with 
PYSPARK_PYTHON=python3 ../test/code/spark-2.3.1-bin-hadoop2.7/bin/spark-submit ./pet-classification-spark.py -d features -c1 keeshond -i 100
"""
sc = SparkContext(appName="pet-classification")
spark = SparkSession.builder.getOrCreate()

def parse_arguments():
    """ Retrieving arguments
    Parsing argument with argparse module
    """
    parser = argparse.ArgumentParser(description='Process pet classification...', 
                                     prog='pet-classification-spark.py')
    parser.add_argument("-d", "--directory", help="""datasets features directory""", 
                        required=True)
    parser.add_argument("-v", "--verbose", action='store_true', 
                        help="""Make the application talk!""", required=False)
    parser.add_argument("-f", "--force", action='store_true', 
                        help="""Forcing training model""", required=False)
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

def filter_class(partition):
    for feature in partition:
        if feature.split(', ')[0] in (class1,class2) or class2=='All':
            yield feature

def main():
    #retrieve argument
    args = parse_arguments()
    main_directory = args.directory
    class1 = args.class1
    class2 = args.class2
    force_by_user = args.force
    if args.verbose:
        lg.basicConfig(level=lg.INFO)
        
    #Variables declaration
    result = []
    directory_feature = os.path.join(main_directory, "features", "*.json") 
    nb_training_data_list = args.nb_training_data
    iteration_model_list = args.iteration_model
    
    lg.info('Features directory is %s', directory_feature)
    for iteration_model in iteration_model_list:
        for nb_training_data in nb_training_data_list:
            model_file = 'model_'+class1+'_'+class2+'_'+str(nb_training_data)+'_'+str(iteration_model)
            result_file = 'result_'+class1+'_'+class2+'_'+str(nb_training_data)+'_'+str(iteration_model)+'_'+time.strftime("%Y%m%d%H%M%S")+'.json'
            model_pathname = os.path.join(main_directory, "models", model_file) 
            

            is_model = False
            
            start_time = time.time()
            lg.info('#################### Starting pet-classification ######################')
            lg.info('Class 1 is %s', class1)
            lg.info('Class 2 is %s', class2)
            lg.info('Number of training datas is %s', nb_training_data)
            lg.info('Number of iterations model is %s', iteration_model)
            
            #persist a common rdd which is using by both training and testing datas
            common_rdd = sc.textFile(directory_feature, minPartitions=4)\
                           .filter(lambda line: line.split(', ')[0] in (class1, class2) or class2 == 'All')\
                           .persist()    

            #Loading model if exists
            if is_model and not force_by_user:
                model = SVMModel.load(sc, model_pathname)
                lg.info('Found and load recorded model %s', model_file)
            else: 
                lg.info('No recorded model found')
                #create training rdd and train model if no model found or force
                train_data_rdd = common_rdd.filter(lambda line: int(line.split(',')[1]) <= nb_training_data)\
                                           .map(lambda line: Row(label=0.0, features=line.split(', ')[2:]) 
                                                if line.split(', ')[0] == class1 
                                                else Row(label=1.0, features=line.split(', ')[2:]))\
                                           .map(lambda line: LabeledPoint(line.label, line.features))
                
                lg.info('%s features for training datas', train_data_rdd.count())
                lg.info('Start to training model')
                model = SVMWithSGD.train(train_data_rdd, iterations=iteration_model)
                lg.info('Training model terminated')
            
            training_time = time.time()
            training_duration = training_time - start_time
            #Create testing rdd
            test_data_rdd = common_rdd.filter(lambda line: int(line.split(', ')[1]) > nb_training_data)\
                              .map(lambda line: Row(label=0.0, features=line.split(', ')[2:]) 
                                   if line.split(', ')[0] == class1 
                                   else Row(label=1.0, features=line.split(', ')[2:]))\
                              .map(lambda row: LabeledPoint(row.label, row.features))
            lg.info('%s features for test datas', test_data_rdd.count())
            
            # Evaluating the model on training data
            predictions = test_data_rdd.map(lambda row: (row.label, float(model.predict(row.features))))
            train_error = predictions.filter(lambda lp: lp[0] != lp[1]).count() \
                                             / float(predictions.count())
            lg.info('Test Error : %s', str(train_error))
            end_time = time.time()
            duration = end_time - start_time
            lg.info('Duration %s', str(duration))
            prediction_duration = end_time - training_time
            # #Save and dump result on S3
            result = {"class1" : class1, 
                           "class2" : class2, 
                           "nb_training_data" : nb_training_data,
                           "error" : train_error,
                           "iteration_model" : iteration_model, 
                           "total_duration" : duration,
                           "training_duration" : training_duration,
                           "prediction_duration" : prediction_duration}
            

            with open(result_file, 'w') as result_file:
                json.dump(result, result_file)
    
    lg.info('#################### Ending pet-classification ######################')
    input("press ctrl+c to exit")

if __name__ == "__main__":
    main()
