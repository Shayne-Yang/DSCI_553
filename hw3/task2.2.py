from pyspark import SparkContext, SparkConf
import time
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
import json
import sys


# get the feature columns
def get_features(x, type,
                 u_temp={"usr_avg": 3.75},
                 b_temp={"bns_avg": 3.75}  # Calculate the average level to impute the missing value when there's
                 # no enough data
                 ):
    rst = {}
    if len(x) <= 0:
        return u_temp if type == "usr" else b_temp  # return a average level if there's all missing value
    ary = pd.Series([float(i[1]) for i in x])
    rst[f"{type}_avg"] = ary.mean()
    return rst


# generate user/business dictionary
def gen_columns(u, b,u_dic,b_dic):
    v1 = u_dic.get(u, [])
    v2 = b_dic.get(b, [])
    v1 = [i for i in v1 if i[0] != b]
    v2 = [i for i in v2 if i[0] != u]
    d1 = get_features(v1, "usr")
    d2 = get_features(v2, "bns")
    d3 = dict(d1, **d2)
    d0 = {"user_id": u, "business_id": b}
    return dict(d0, **d3)

if __name__ == '__main__':
    begin = time.time()
    # folder = '/Users/shayne/Documents/study/Master/USC/Semester3/DSCI553/hw3'
    # test_file = 'yelp_train.csv'
    # output_file = 'yelp_val.csv'

    folder = sys.argv[1]
    test_file = sys.argv[2]
    output_file = sys.argv[3]

    conf = SparkConf().setMaster("local") \
        .setAppName("hw_3_task2.2") \
        .set("spark.executor.memory", "15g") \
        .set('spark.executor.cores', '10') \
        .set("spark.driver.memory", "20g") \
        .set('spark.executor.instances', '20')

    sc = SparkContext(conf=conf)
    sc.setLogLevel("WARN")

    # use 20 partitions as the default one
    num_partitions = 5

    ## 1. Data manipulation

    # import tip
    tip = sc.textFile(folder + "/tip.json").map(lambda x: json.loads(x)).map(lambda x: (x['business_id'], x['likes'])) \
        .reduceByKey(lambda x, y: x + y).collectAsMap()

    # import business
    business = sc.textFile(folder + "/business.json").map(lambda x: json.loads(x)) \
        .map(lambda x: (x["business_id"], (x["stars"], x["review_count"]))).collectAsMap()

    # import photo
    photo = sc.textFile(folder + "/photo.json") \
        .map(lambda x: json.loads(x)).map(lambda x: (x['business_id'], 1)).reduceByKey(lambda x, y: x + y).collectAsMap()

    # import check
    check = sc.textFile(folder + "/checkin.json").map(lambda x: json.loads(x)) \
        .map(lambda x: (x["business_id"], sum(list(x["time"].values())))).collectAsMap()

    u_temp = {"usr_avg": 3.75}
    b_temp = {"bns_avg": 3.75}

    # import train data: business:(user, score)
    train = sc.textFile(folder + "/yelp_train.csv").map(lambda x: x.split(",")).map(lambda x: (x[1], (x[0], x[2])))
    try:
        header = train.first()
    except:
        header = train.first()

    # delete the header
    data = train.filter(lambda x: x != header)
    b_data = data.map(lambda x: (x[0], x[1])).groupByKey().mapValues(lambda x: list(x))
    b_dic = b_data.collectAsMap()

    # delete the header
    # generate user dict: user:(business,score)
    train = sc.textFile(folder + "/yelp_train.csv").map(lambda x: x.split(",")).map(lambda x: (x[0], (x[1], x[2])))
    header = train.first()
    data = train.filter(lambda x: x != header)
    u_data = data.map(lambda x: (x[0], x[1])).groupByKey().mapValues(lambda x: list(x))
    u_dic = u_data.collectAsMap()

    train_df = data.repartition(num_partitions).map(lambda x: (gen_columns(x[0], x[1][0],u_dic,b_dic), x[1][1])).collect()
    bus_related = data.repartition(num_partitions).map(
        lambda x: {"business": business.get(x[1][0], None), "tip": tip.get(x[1][0], None), \
                   "photo": photo.get(x[1][0], None), "check": check.get(x[1][0], None)}).collect()

    init = pd.DataFrame(bus_related)
    init["stars"] = init["business"].transform(lambda x: x[0])
    init["review_cnt"] = init["business"].transform(lambda x: x[1])
    init.drop("business", axis=1, inplace=True)

    # all the features below
    X = [i[0] for i in train_df]
    train_X = pd.concat([pd.DataFrame(X), init], axis=1)
    train_x = train_X[['usr_avg', 'bns_avg', 'tip', 'photo', 'check', 'stars', 'review_cnt']]
    # all the y data below
    y = [float(i[1]) for i in train_df]
    train_y = pd.Series(y)

    train2 = sc.textFile(test_file).map(lambda x: x.split(",")).map(lambda x: (x[0], x[1]))
    header2 = train2.first()
    data2 = train2.filter(lambda x: x != header2)

    bus_related2 = data2.repartition(num_partitions).map(
        lambda x: {"business": business.get(x[1], None), "tip": tip.get(x[1], None), \
                   "photo": photo.get(x[1], None), "check": check.get(x[1], None)}).collect()

    init2 = pd.DataFrame(bus_related2)
    init2["stars"] = init2["business"].transform(lambda x: x[0])
    init2["review_cnt"] = init2["business"].transform(lambda x: x[1])
    init2.drop("business", axis=1, inplace=True)

    test_df = data2.repartition(num_partitions).map(lambda x: gen_columns(x[0], x[1],u_dic,b_dic))
    test = test_df.collect()

    X = [i for i in test]
    test_x = pd.concat([pd.DataFrame(X), init2], axis=1)
    output_df = test_x[['user_id', 'business_id']]
    test_x = test_x[['usr_avg', 'bns_avg', 'tip', 'photo', 'check', 'stars', 'review_cnt']]

    ## 2. data modeling

    model = XGBRegressor(
        max_depth=5,
        min_child_weight=1,
        subsample=0.7,
        colsample_bytree=0.8,
        gamma=0,
        reg_alpha=1,
        reg_lambda=0,
        learning_rate=0.01,
        n_estimators=500
    )

    model.fit(train_x, train_y)

    prd_array = list(float(i) for i in model.predict(test_x).tolist())
    output_df["prediction"] = prd_array

    output_df.to_csv(output_file, index=False)

    end = time.time()
    print(f"the execution time is {end - begin}")