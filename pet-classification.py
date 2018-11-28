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
Use with PYSPARK_PYTHON=python3 ../test/code/spark-2.3.1-bin-hadoop2.7/bin/spark-submit 
./pet-classification.py -d features -c 1vsAll -r 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 -i 30 50 70 100 150 -v
"""
sc = SparkContext()
spark = SparkSession.builder.getOrCreate()


def parse_arguments():
    """ Retrieving arguments
    Parsing argument with argparse module
    """
    parser = argparse.ArgumentParser(description='Process pet classification...', 
                                     prog='pet-classification.py')
    parser.add_argument("-d", "--directorie", help="""datasets directorie""", 
                        required=True)
    parser.add_argument("-v", "--verbose", action='store_true', 
                        help="""Make the application talk!""", required=False)
    parser.add_argument("-r", "--random_split", default=0.1, nargs='+', 
                        help="""random split percent""", type=float, required=False)
    parser.add_argument("-c", "--classification_type", 
                        help="""1vs1 or 1vsAll""", choices=['1vs1', '1vsAll'],
                        required=True)
    parser.add_argument("-i", "--iteration_model", 
                        help="""Number of iteration for SVMWithSGD training model""", 
                        required=False,
                        nargs='*',
                        default=100,
                        type=int
                        )
    return parser.parse_args()

def extract_class(filename):
    """
    Extracting classname based on filename 
    input : german_shorthaired_59.jpg.json
    output : German_Shorthaired
    """
    return re.sub(r'[0-9]', '', filename.split('/')[-1])[:-9].strip('_').title()

def load_features(directorie, class1, class2):
    """ 
    Make a dictionnary of features by class
    Each classe key contains the list of their features
    """
    classes_features = []
    features = []
    for filename in os.listdir(directorie):
        #Extract class from filename wich is the dictionnarie key
        current_class = extract_class(filename)
        if current_class in (class1, class2):
            #features = json.load(open(directorie+"/"+filename, "r"))
            features = [float(feature) for feature in json.load(open(directorie+"/"+filename, "r"))]
            features.insert(0, current_class)
            #update classe list
            classes_features.append(features)
        if class2 == "All" and current_class != class1:
            features = [float(feature) for feature in json.load(open(directorie+"/"+filename, "r"))]
            features.insert(0, "All")
            #update classe list
            classes_features.append(features)
    lg.info('%s features availables', len(classes_features))
    return classes_features

def split_data(classe_features, nb_classes_features):
    """
    splitting data into 2 lists of features lists
    For each features lists, first element is classe and others are features
    deprecated : replaced by pyspark randomSplit 
    """
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

def load_dataframe(classe_feature_list, class1, class2):
    """
    Create a dataframe with param list and filter
    keep class 1 and class 2
    """
    rdd = sc.parallelize(classe_feature_list,4)\
            .filter(lambda feature_list: (feature_list[0] == class1 or feature_list[0] == class2))\
            .map(lambda feature_list: Row(label=feature_list[0],
                                          features=feature_list[1:]))
    return spark.createDataFrame(rdd)

def create_labeledpoint(dataframe):
    """
    Create a labeledPoint with dataframe passed in parameter
    Datalabeledpoint needs numeric index row
    """
    datalabeledpoint = dataframe.rdd.map(lambda row: LabeledPoint(row.label_index, row.features))
    return datalabeledpoint

def choose_random_classes(directorie, classification_type):
    """
    return 2 classes based on features present in feature's directorie
    classe2 depends on classification type
    """
    classes_list = []
    for filename in os.listdir(directorie):
        current_class = extract_class(filename)
        if current_class not in classes_list:
            classes_list.append(current_class)
    class1 = classes_list.pop(randint(0, len(classes_list)-1))
    class2 = 'All'
    if classification_type == '1vs1':
        class2 = classes_list.pop(randint(0, len(classes_list)-1))
    lg.info('Class 1 is %s', class1)
    lg.info('Class 2 is %s', class2)
    return class1, class2    
    
def choose_random_classes_v2(directorie, classification_type):
    """
    wholeTextFiles preserves the relation between data 
    and the files that contained it, by loading the data 
    into a PairRDD with one record per input file. 
    The record will have the form (fileName, fileContent)
    """
    classes_list = sc.wholeTextFiles(path=directorie, minPartitions=4)\
                    .map(lambda file: file[0])\
                    .map(extract_class)\
                    .distinct()\
                    .collect()
    class1 = classes_list.pop(randint(0, len(classes_list)-1))
    class2 = 'All'
    if classification_type == '1vs1':
        class2 = classes_list.pop(randint(0, len(classes_list)-1))
    lg.info('Class 1 is %s', class1)
    lg.info('Class 2 is %s', class2)
    return class1, class2

def main():
    args = parse_arguments()
    result = []
    result_log = []
    if args.verbose:
        lg.basicConfig(level=lg.INFO)
    try:
        directorie = args.directorie
        classification_type = args.classification_type
        split_percent_list = args.random_split
        iteration_model_list = args.iteration_model
        if not os.path.exists(directorie):
            raise FileNotFoundError('directorie {} does not exist'.format(directorie))
    except FileNotFoundError as no_directorie:
        lg.critical(no_directorie)
    else:  
        lg.info('#################### Starting pet-classification ######################')
        lg.info('Choosing %s classification', classification_type)
        step = 0
        label_indexer = StringIndexer(inputCol="label", outputCol="label_index") 
        class1, class2 = choose_random_classes(directorie, classification_type)
        class_feature = load_features(directorie, class1, class2)
        datatrain = load_dataframe(class_feature, class1, class2)
        for split_percent in split_percent_list:
            for iteration_model in iteration_model_list:
                
                lg.info('#################### Starting step %s ####################', step+1)
                lg.info('Random split is %s', split_percent)
                lg.info('Number of iterations model is %s', iteration_model)
                
                # class_feature.clear()
                
                #split dataframe into training datas and testing datas
                (training_data, test_data) = datatrain.randomSplit([1-split_percent, split_percent])
                # training_data.persist()
                # test_data.persist()

                # add lable index on train and test datas - requirement for datalabeledpoint use  
                label_index_model = label_indexer.fit(training_data) 
 
                training_data = label_index_model.transform(training_data)
                test_data = label_index_model.transform(test_data)

                #Create datalabledpoint for train & test datas
                training_datalabeledpoint = create_labeledpoint(training_data)
                test_datalabeledpoint = create_labeledpoint(test_data)

                # Build the model
                model = SVMWithSGD.train(training_datalabeledpoint, iterations=iteration_model)

                # # # Evaluating the model on testing data
                predictions = test_datalabeledpoint.map(lambda row: (row.label, float(model.predict(row.features))))
            
                train_error = predictions.filter(lambda lp: lp[0] != lp[1]).count() \
                                                 / float(predictions.count())

                lg.info('Test Error ================>%s', str(train_error))
                lg.info('##################### Ending step %s #####################', step+1)
                step += 1
                result_log.append([iteration_model,split_percent,train_error])
                result.append({"step" : step, "class1" : class1, "class2" : class2,
                               "iteration_model" : iteration_model, 
                               "split_percent" : split_percent, "error" :  train_error})

        print(result_log)
        with open('result.json', 'w') as result_file:
            json.dump(result, result_file)
    finally:
        lg.info('#################### Ending pet-classification ######################')
        input("press ctrl+c to exit")

if __name__ == "__main__":
    main()
