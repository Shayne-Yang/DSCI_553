from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from graphframes import *
import sys
import time

# input_file = 'ub_sample_data.csv'
# output_file = 'test1.txt'
# threshold = 7

if __name__ == '__main__':
    begin = time.time()
    threshold = sys.argv[1]
    input_file = sys.argv[2]
    output_file = sys.argv[3]

    conf = SparkConf().setMaster("local") \
        .setAppName("hw4_task1") \
        .set("spark.executor.memory", "15g") \
        .set('spark.executor.cores', '10') \
        .set("spark.driver.memory", "20g") \
        .set('spark.executor.instances', '20')

    sc = SparkContext(conf=conf)
    sc.setLogLevel("WARN")

    # create a SQL object
    spark = SparkSession.builder \
        .appName('build_a_df.com') \
        .getOrCreate()

    rdd = sc.textFile(input_file) \
        .map(lambda x: x.split(',')).cache()

    header = rdd.first()

    # get user rdd: (user_id,set(business_id1...))
    user_rdd = rdd.filter(lambda x: x != header) \
        .map(lambda x: (x[0], x[1])) \
        .groupByKey() \
        .mapValues(lambda x: set(x))

    # get the user_id edge
    # ((user1,business_list1),(user2,business_list2)) --> ((user1,user2),len) --> filter --> (user1,user2)
    # threshold = 7
    edge_rdd_du = user_rdd.cartesian(user_rdd) \
        .map(lambda x: ((x[0][0], x[1][0]), len(x[0][1] & x[1][1]))) \
        .filter(lambda x: x[1] >= threshold) \
        .filter(lambda x: x[0][0] != x[0][1]) \
        .map(lambda x: (x[0][0], x[0][1])).distinct()

    edge_rdd = edge_rdd_du.filter(lambda x: x[0] < x[1])
    # get the user_id vertex : [userid]
    # only include those node with edge
    vertex_rdd = edge_rdd_du.map(lambda x: x[0]).distinct()

    # with open('vertex.txt','w+') as f:
    #     for vertex in vertex_rdd.collect():
    #         f.write(vertex)
    #         f.write('\n')
    vertex_schema = 'id'
    edge_schema = ['src', 'dst']

    vertex_df = spark.createDataFrame(vertex_rdd, "string").toDF(vertex_schema)
    edge_df = spark.createDataFrame(edge_rdd, edge_schema)
    k = edge_rdd.collect()
    g = GraphFrame(vertex_df, edge_df)

    # use Label Propagation
    result = g.labelPropagation(maxIter=5)

    result_rdd = result.rdd.map(tuple) \
        .map(lambda x: (x[1], x[0])) \
        .groupByKey() \
        .map(lambda x: sorted(x[1]))

    communities = sorted(result_rdd.collect(), key=lambda x: (len(x), x))

    with open(output_file, 'w+') as f:
        for community in communities:
            community_ls = ["'" + word + "'" for word in community]
            f.write(','.join(community_ls))
            f.write('\n')

    print(f'the execution time is {time.time() - begin}')
