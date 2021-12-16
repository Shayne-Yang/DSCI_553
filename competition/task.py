"""
Method Description:
I used XGBRegressor and did a lot of feature engineering to improve my recommendation system.
I basically only use model-based recommendation system. A hybrid recommendation system is considered just before, but it did not
perform very well.

Error Distribution:
>=0 and <1: 102347
>=1 and <2: 32748
>=2 and <3: 6159
>=3 and <4: 790
>=4: 0

RMSE:
0.97715

Execution Time:
1200s
"""



from pyspark import SparkContext, SparkConf
import time
import pandas as pd
from xgboost import XGBRegressor
import numpy as np
import json
import math
import sys


def feature_engineer(feature, type):
    """
    :param feature:The vector we want to get its feature collection
    :param type: The type of estimation we need
    :return: the feature output
    """
    rst = {}
    if len(feature) <= 0:
        return user_general if type == "usr" else bu_general
    array_list = pd.Series([float(i[1]) for i in feature])
    rst[f"{type}_avg"] = array_list.mean()
    rst[f"{type}_std"] = array_list.std()
    rst[f"{type}_kurt"] = array_list.kurt()
    rst[f"{type}_skew"] = array_list.skew()
    rst[f"{type}_max"] = array_list.max()
    rst[f"{type}_min"] = array_list.min()
    return rst


def column_generation(user, business_id):
    """

    :param user:the user_dic
    :param business_id: the business_dic
    :return: the engineered features dic
    """
    vector_1 = u_dic.get(user, [])
    vector_2 = b_dic.get(business_id, [])
    vector_1 = [i for i in vector_1 if i[0] != business_id] # get this customer's remaining business_id
    vector_2 = [i for i in vector_2 if i[0] != user] # get this business's remaining customers
    dic1 = feature_engineer(vector_1, "usr")
    dic2 = feature_engineer(vector_2, "bns")
    dic3 = dict(dic1, **dic2)
    dic0 = {"user_id": user, "business_id": business_id}
    return dict(dic0, **dic3)


def cnt_label(label_list):
    temp = ['drink', 'food', 'inside', 'menu', 'outside']
    record = {i: 0 for i in temp}
    for i in temp:
        record[i] = len([j for j in label_list if j == i])
    return record


def get_mean(raw_list):
    """if it has the value, then get its average. Else it will return the general average which is calculated before"""
    raw_list = [float(i[1]) for i in raw_list]
    if len(raw_list) == 0:
        return 3.7505 #all of average_score
    else:
        return sum(raw_list) / len(raw_list)

def combine_statistics(user,business_id,score,user_list,business_list):
    try:
        u_raw_list = user_list.remove(score)
    except:
        print(score)
        print(user_list)
    user_stat_dic = get_statistics(u_raw_list, temp=user_general, label='usr')
    bu_raw_list = business_list.remove(score)
    bu_stat_dic = get_statistics(bu_raw_list, temp=bu_general, label='bus')
    initial = {'user_id':user,'business_id':business_id,'y':score}
    return dict(initial,**user_stat_dic,**bu_stat_dic)

def get_statistics(raw_list,temp=None,label='usr'):
    np_list = pd.Series(raw_list)
    stat_dict = {}
    if raw_list and len(raw_list) > 0:
        stat_dict[f'{label}_avg'] = np_list.mean()
        stat_dict[f'{label}_std'] = np_list.std()
        stat_dict[f'{label}_kurt'] = np_list.kurt()
        stat_dict[f'{label}_skew'] = np_list.skew()
        stat_dict[f'{label}_max'] = np_list.max()
        stat_dict[f'{label}_min'] = np_list.min()
        return stat_dict
    else:
        return temp

def friends_manipulate(input):
    if input == "None":
        return 0
    else:
        return len(input.split(","))

def merge_data(train_data, feature_list, users):
    for features in feature_list:
        temp_data = pd.DataFrame(features)
        train_data = pd.merge(train_data, temp_data, on="business_id", how="left")

    temp_data = pd.DataFrame(users)
    train_data = pd.merge(train_data, temp_data, on="user_id", how="left")

    return train_data

if __name__ == "__main__":
    conf = SparkConf().setMaster("local") \
        .setAppName("Competition") \
        .set("spark.executor.memory", "15g") \
        .set('spark.executor.cores', '10') \
        .set("spark.driver.memory", "20g") \
        .set('spark.executor.instances', '20')

    sc = SparkContext(conf=conf)
    sc.setLogLevel("WARN")

    file_name = sys.argv[1]
    test_file = sys.argv[2]
    output_file = sys.argv[3]

    # file_name = '/Users/shayne/Documents/study/Master/USC/Semester3/DSCI553/competition'
    # test_file = 'yelp_val.csv'
    # output_file = 'output.txt'
    num_partitions = 20  # initialize the number of partitions

    # because we need the statistical for creating more variables, we did some calculate before
    # Caculate the whole summary before:
    user_general = {"usr_avg": 3.75117,"usr_std": 1.03238,"usr_kurt": 0.33442,"usr_skew": -0.70884
        ,'usr_max':5,'usr_min':1}

    bu_general = {"bns_avg": 3.75088,"bns_std": 0.990780,"bns_kurt": 0.48054,"bns_skew": -0.70888
        ,'bns_max': 5, 'bns_min': 1}

    # 1. manipulate the training data
    train = sc.textFile(file_name + "/yelp_train.csv") \
        .map(lambda x: x.split(","))

    header = train.first()
    data = train.filter(lambda x: x != header)  # delete the header

    u_rdd = data.map(lambda x: (x[0], (x[1], x[2]))) \
        .map(lambda x: (x[0], x[1])) \
        .groupByKey() \
        .mapValues(lambda x: list(x))
    u_dic = u_rdd.collectAsMap()

    b_rdd = data.map(lambda x: (x[1], (x[0], x[2]))) \
        .map(lambda x: (x[0], x[1])) \
        .groupByKey() \
        .mapValues(lambda x: list(x))
    b_dic = b_rdd.collectAsMap()
    # 2. Manipulate the feature data

    # business --- general data
    business = sc.textFile(file_name + "/business.json").map(lambda x: json.loads(x)) \
        .map(lambda x: (x["business_id"], (x["stars"], x["review_count"], x['latitude'], x['longitude'],
                                           x['is_open'])))\
        .map(lambda bu:{"business_id": bu[0], "b_stars": bu[1][0], "b_review_count": bu[1][1], 'latitude': bu[1][2],
                 'longitude': bu[1][3], 'is_open': bu[1][4]})\
        .collect()
    # business = [{"business_id": bu[0], "b_stars": bu[1][0], "b_review_count": bu[1][1], 'latitude': bu[1][2],
    #              'longitude': bu[1][3], 'is_open': bu[1][4]} for bu in business0]

    # business data --- attributes data
    business_raw = sc.textFile(file_name + "/business.json").map(lambda x: json.loads(x)).collect()
    attribute_dic = dict()
    attribute_cnt = dict()
    business_len = len(business_raw)
    for row in business_raw:
        if row.get('attributes',None):
            sub_dic = row['attributes']
            for j in sub_dic.keys():
                attribute_dic.setdefault(j, set())
                attribute_dic[j].add(sub_dic[j])
                attribute_cnt.setdefault(j, 0)
                attribute_cnt[j] += 1 / business_len  # compute the percentage

    fea_needs = [key for key,value in attribute_dic.items() if len(value) <= 10 and attribute_cnt[key] >= 0.2]
    temp_dic = {i: None for i in fea_needs}
    business_attribute = []
    for row in business_raw:
        bus_record = temp_dic.copy()
        bus_record["business_id"] = row["business_id"]
        if row.get('attributes',None):
            sub_dic = row['attributes']
            for j in sub_dic.keys():
                if j in fea_needs:
                    bus_record[j] = sub_dic[j]
            business_attribute.append(bus_record.copy())
    del business_raw

    # tips
    tips = sc.textFile(file_name + "/tip.json") \
        .map(lambda x: json.loads(x)) \
        .map(lambda x: (x['business_id'], (x['likes'], 1))) \
        .reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1]))\
        .map(lambda x: {"business_id": x[0], "likes_sum": x[1][0], "likes_cnt": x[1][1]})\
        .collect()


    # photos
    photo = sc.textFile(file_name + "/photo.json").map(lambda x: json.loads(x)).map(
        lambda x: (x['business_id'], [x["label"]])).reduceByKey(lambda x, y: x + y).collect()
    photos = []

    for row in photo:
        rcd = cnt_label(row[1])
        s = sum(rcd.values())
        rcd["photo_sum"] = s
        rcd["business_id"] = row[0]
        photos.append(rcd.copy())

    del photo

    # checks
    checks = sc.textFile(file_name + "/checkin.json").map(lambda x: json.loads(x)) \
        .map(lambda x: (x["business_id"], (sum(list(x["time"].values())), len(list(x["time"].values())))))\
        .map(lambda x:{'business_id':x[0],'slots':x[1][0],'customers':x[1][1]})\
        .collect()



    def check_fea(obj):
        l = list(obj.items())
        rst = dict()
        for i, j in l:
            rst.setdefault(i[:3], 0)
            rst[i[:3]] += j
        return rst


    # users
    user_raw = sc.textFile(test_file).map(lambda x: x.split(",")).map(lambda x: (x[0], x[1]))
    test_header = user_raw.first()
    test_2 = user_raw.filter(lambda x: x != test_header)
    user_index = set(test_2.map(lambda x: x[0]).distinct().collect() + list(u_dic.keys()))
    users = sc.textFile(file_name + "/user.json").map(lambda x: json.loads(x)).filter(
        lambda x: x["user_id"] in user_index).collect()
    for u in users:
        del u["name"]
        u["friends"] = friends_manipulate(u["friends"])
        u["elite"] = friends_manipulate(u["elite"])

    # 3. merge all of dataframe to get the whole data

    user_agg_rdd = data.map(lambda x:(x[0],float(x[2]))).groupByKey()
    bu_agg_rdd = data.map(lambda x:(x[1],float(x[2]))).groupByKey()

    # join the data to get the statistical summary
    # (user_id,(bu_id,score)) -> (user_id,((bu_id,score),(score_list))
    # -> (bu_id,(user_id,score,score_list)) -->(bu_id,((user_id,score,score_list),bu_score_list))
    # -> (user_id,bu_id,score,score_list,bu_score_list)

    train_rdd = data.map(lambda x:(x[0],(x[1],float(x[2]))))\
        .join(user_agg_rdd)\
        .map(lambda x:(x[1][0][0],(x[0],x[1][0][1],x[1][1])))\
        .join(bu_agg_rdd)\
        .map(lambda x:(x[1][0][0],x[0],x[1][0][1],x[1][0][2],x[1][1]))\
        .repartition(50)\
        .map(lambda x:combine_statistics(x[0],x[1],x[2],list(x[3]),list(x[4])))\
        .collect()




    feature_list = [business,business_attribute,tips,photos,checks]
    train_data = merge_data(pd.DataFrame(train_rdd),feature_list,users)


    for col in pd.DataFrame(business_attribute).columns: # except the business_id
        if col == 'business_id':
            continue
        else:
            temp_data = pd.get_dummies(train_data[col].fillna("Unknown"), drop_first=True)
            new_cols = [col + i for i in temp_data.columns]
            temp_data.columns = new_cols
            train_data.drop(col, axis=1, inplace=True)
            train_data = pd.concat([train_data, temp_data], axis=1)

    # dealing with missing value
    train_data['bns_avg'].fillna(bu_general['bns_avg'],inplace=True)
    train_data['bns_std'].fillna(bu_general['bns_std'], inplace=True)
    train_data['bns_kurt'].fillna(bu_general['bns_kurt'], inplace=True)
    train_data['bns_skew'].fillna(bu_general['bns_skew'], inplace=True)
    train_data['bns_max'].fillna(bu_general['bns_avg'], inplace=True)
    train_data['bns_min'].fillna(bu_general['bns_avg'], inplace=True)
    train_data.fillna(0, inplace=True)
    # time data
    train_data["history"] = (2021 - train_data["yelping_since"].transform(lambda x: int(x[:4]))) * 12 + train_data["yelping_since"].transform(lambda x: int(x[5:7]))  # get the momth data
    # Do the same thing to the test data
    test = sc.textFile(test_file).map(lambda x: x.split(","))
    test_header = test.first()
    test_2 = test.filter(lambda x: x != test_header)

    test_df = test_2.repartition(num_partitions).map(lambda x: column_generation(x[0], x[1])).collect()
    test_data = merge_data(pd.DataFrame(test_df),feature_list,users)

    # we only want those features that train_data has

    valid_cols = train_data.columns
    for col in pd.DataFrame(business_attribute).columns:
        if col == 'business_id':
            continue
        else:
            temp_data = pd.get_dummies(test_data[col].fillna("Unknown"))
            cols = [i for i in temp_data.columns if col + i in valid_cols]
            temp_data = temp_data[cols]
            temp_data.columns = [col + i for i in temp_data.columns]
            test_data.drop(col, axis=1, inplace=True)
            test_data = pd.concat([test_data, temp_data], axis=1)


    # dealing with missing value
    test_data['bns_avg'].fillna(bu_general['bns_avg'], inplace=True)
    test_data['bns_std'].fillna(bu_general['bns_std'], inplace=True)
    test_data['bns_kurt'].fillna(bu_general['bns_kurt'], inplace=True)
    test_data['bns_skew'].fillna(bu_general['bns_skew'], inplace=True)
    test_data['bns_max'].fillna(bu_general['bns_avg'], inplace=True)
    test_data['bns_min'].fillna(bu_general['bns_avg'], inplace=True)
    test_data.fillna(0, inplace=True)

    # time data
    test_data["history"] = (2021 - test_data["yelping_since"].transform(lambda x: int(x[:4]))) * 12 +\
                           test_data["yelping_since"].transform(lambda x: int(x[5:7]))

    # 4. XGboost
    train_data["y"] = train_data["y"].astype("float")
    train_cols = train_data.columns.difference(["y", "business_id", "user_id", "yelping_since"])
    train_x = train_data[train_cols]
    train_y = train_data["y"]
    # print('step2')
    model = XGBRegressor(
        max_depth=5,
        min_child_weight=1,
        subsample=0.6,
        colsample_bytree=0.6,
        gamma=0,
        reg_alpha=1,
        reg_lambda=0,
        learning_rate=0.05,
        n_estimators=800
    )
    model.fit(train_x, train_y, early_stopping_rounds=20, eval_set=[(train_x, train_y)],verbose=50)  # ,(test_x, test_y)

    preds = pd.concat([test_data[["user_id", "business_id"]], pd.DataFrame(model.predict(test_data[train_cols]))], axis=1)
    preds.columns = ["user_id", "business_id", "prediction"]

    preds.to_csv(output_file, index=False)

    # validation
    val_data = pd.read_csv('yelp_val.csv')
    valid_set = val_data.merge(preds,on=['user_id','business_id'])
    valY = np.array(valid_set['stars'])
    prd_array = np.array(valid_set['prediction'])

    def RMSE(np1,np2):
        return math.sqrt(sum((np1-np2)**2)/len(np1))

    print(RMSE(valY,prd_array))
