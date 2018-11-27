#! /usr/bin/env python3
# coding: utf-8

import argparse
import re
import os
import logging as lg
import json
from random import randint
from pyspark.sql import Row
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.mllib.classification import SVMWithSGD
from pyspark.mllib.regression import LabeledPoint
from pyspark.ml.feature import StringIndexer

"""
Use with PYSPARK_PYTHON=python3 ../test/code/spark-2.3.1-bin-hadoop2.7/bin/spark-submit 
./pet-classification.py -d features -n 10
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
    parser.add_argument("-n", "--nb_features", 
                        help="""Number of training datas""", type=int, required=True)
    parser.add_argument("-c", "--classification_type", 
                        help="""1vs1 or 1vsAll""", choices=['1vs1', '1vsAll'],
                        required=True)
    return parser.parse_args()

def extract_class(filename):
    """
    Extracting classname based on filename 
    input : german_shorthaired_59.jpg.json
    output : German_Shorthaired
    """
    return re.sub(r'[0-9]', '', filename.split('/')[-1])[:-9].strip('_').title()

def load_features(directorie):
    """ 
    Make a dictionnary of features by class
    Each classe key contains the list of their features
    """
    classes_features = []
    features = []
    for filename in os.listdir(directorie):
        #Extract class from filename wich is the dictionnarie key
        current_class = extract_class(filename)
        features = json.load(open(directorie+"/"+filename, "r"))
        features.insert(0, current_class)
        #update classe list
        classes_features.append(features)
    lg.info('%s features availables', len(classes_features))
    return classes_features

def split_data(classe_features, nb_classes_features):
    """
    splitting data into 2 lists of features lists
    For each features lists, first element is classe and others are features
    deprecated : use randomSplit pyspark
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
    rdd = sc.parallelize(classe_feature_list)\
            .filter(lambda feature_list: (feature_list[0] == class1 or feature_list[0] == class2))\
            .map(lambda feature_list: Row(label=feature_list[0],
                                          features=feature_list[1:]))
    return spark.createDataFrame(rdd)

def create_labeledPoint(dataframe):
    datalabeledpoint = dataframe.rdd.map(lambda row: LabeledPoint(row.label_index,row.features))
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
    lg.info('Class 1 is  %s', class1)
    lg.info('Class 2 is  %s', class2)
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
    if args.verbose:
        lg.basicConfig(level=lg.INFO)
    try:
        directorie = args.directorie
        classification_type = args.classification_type
        if not os.path.exists(directorie):
            raise FileNotFoundError('directorie {} does not exist'.format(directorie))
    except FileNotFoundError as no_directorie:
        lg.critical(no_directorie)
    else:   
        lg.info('#################### Starting pet-classification ######################')
        lg.info('Choosing %s classification', classification_type)
        class1, class2 = choose_random_classes(directorie, classification_type)
        classe_feature = load_features(directorie)
        datatrain = load_dataframe(classe_feature, class1, class2)
        #split datas into training datas and testing datas
        (trainingData, testData) = datatrain.randomSplit([0.7, 0.3])
        trainingData.show(truncate=True)
        # # classe_feature_training.clear()
        # test_df = load_dataframe(classe_feature_test, class1, class2)
        # # classe_feature_test.clear()

        # label_indexer = StringIndexer(inputCol="label", outputCol="label_index")
        # label_indexer_transformer = label_indexer.fit(training_df)
        # training_df = label_indexer_transformer.transform(training_df)

        # #training_df.show(truncate=True)
        # training_datalabeledpoint = create_labeledPoint(training_df)
        # test_datalabeledpoint = create_labeledPoint(test_df)
        # # # Build the model
        # model = SVMWithSGD.train(training_datalabeledpoint,iterations=30)

        # # # Evaluating the model on training data
        # labelsAndPreds = test_datalabeledpoint.map(lambda row:(row.label,float(model.predict(row.features))))

        
        # trainErr = labelsAndPreds.filter(lambda lp: lp[0] != lp[1]).count() / float(labelsAndPreds.count())
        # # print(labelsAndPreds.take(10))
        #print("Test Error = " + str(trainErr))

        #input("press ctrl+c to exit")


        
    finally:
        lg.info('#################### Ending pet-classification ######################')


if __name__ == "__main__":
    main()
