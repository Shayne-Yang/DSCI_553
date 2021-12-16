from pyspark import SparkContext, SparkConf
import math
import time
import sys

# read the file
def csv_rdd(csvfile):  # read a csv and delete the tile
    rdd = sc.textFile(csvfile) \
        .map(lambda x: x.split(','))
    header = rdd.first()
    rdd = rdd.filter(lambda x: x != header)
    return rdd
#[user_id,business_id,score]

# input: dict1,dict2 --> dict1:{user1:score1...}
def pearson_sililarity(t1,t2,corated_threshold = 40):
    dict1 = dict(t1)
    dict2 = dict(t2)
    all_items = set(dict1.keys()) & set(dict2.keys())
    if len(all_items) <= corated_threshold:# filter pairs which are low corated
        return 0
    else:
        l1_mean = sum(dict1[i] for i in all_items)/len(all_items)
        l2_mean = sum(dict2[i] for i in all_items)/len(all_items)
        dot_product = sum((dict1[i]-l1_mean)*(dict2[i]-l2_mean) for i in all_items)
        l1 = sum((dict1[i]-l1_mean) ** 2 for i in all_items)
        l2 = sum((dict2[i]-l2_mean) ** 2 for i in all_items)
        mod_product = l1*l2
        if mod_product == 0:
            return 0
        else:
            return dot_product/math.sqrt(l1*l2)

# input_row:((b1,scoredict1),(b2,scoredict2))
def yield_pearson_weight(partition):
    for row in partition:
        score = pearson_sililarity(row[0][1],row[1][1])
        if score > 0:
            yield ((row[0][0],row[1][0]),score)

def predict(business,
        user_score, item_relation,b_rate_dict):
    # if user_score == None and item_relation == None:
    #     return 3
    # elif user_score == None:
    #     nominator = sum(b_rate_dict[item]*item_relation[item] for item in item_relation.keys())
    #     denominator = sum(abs(item_relation[item]) for item in item_relation.keys())
    #     return nominator/denominator
    # elif item_relation == None:
    #     return sum(k for k in user_score.values()) / len(user_score)
    # else:
    #     item = set(user_score.keys() & item_relation.keys())
    #     if len(item) != 0:
    #         nominator = sum(user_score[a] * item_relation[a] for a in item)
    #         denominator = sum(abs(item_relation[a]) for a in item)
    #         return nominator / denominator
    #     else:
    #         nominator = sum(b_rate_dict[item] * item_relation[item] for item in item_relation.keys())
    #         denominator = sum(abs(item_relation[item]) for item in item_relation.keys())
    #         return nominator / denominator
    if user_score != None and item_relation != None:
        item = set(user_score.keys() & item_relation.keys())
        if len(item) != 0:
            nominator = sum(user_score[a]*item_relation[a] for a in item)
            denominator = sum(abs(item_relation[a]) for a in item )
            return nominator/denominator
        else:
            return b_rate_dict[business]
    else:
        return b_rate_dict.get(business,3)

if __name__ == '__main__':
    begin = time.time()
    conf = SparkConf().setMaster("local") \
        .setAppName("hw_3_task2.1") \
        .set("spark.executor.memory", "15g") \
        .set('spark.executor.cores', '10') \
        .set("spark.driver.memory", "20g") \
        .set('spark.executor.instances', '20')

    sc = SparkContext(conf=conf)
    sc.setLogLevel("WARN")

    # train_file = 'yelp_train.csv'
    # valid_file = 'yelp_val.csv'
    # test_file = 'yelp_test_in.csv'
    # output_file = 'test2.1.csv'

    train_file = sys.argv[1]
    test_file = sys.argv[2]
    output_file = sys.argv[3]


    train_rdd = csv_rdd(train_file)
    # valid_rdd = csv_rdd(valid_file)
    test_rdd = csv_rdd(test_file)

    input_rdd = train_rdd
    output_rdd = test_rdd # we first use validation set to optimize our model

    b_dict = input_rdd.map(lambda x:x[1])\
        .distinct()\
        .sortBy(lambda x:x)\
        .zipWithIndex()\
        .map(lambda x:(x[0],x[1]))\
        .collectAsMap() #get dict like business_id:index

    b_dict_reverse = {j:i for (i,j) in b_dict.items()}  #get dict like index:business_id

    c_dict = input_rdd.map(lambda x:x[0])\
        .distinct()\
        .sortBy(lambda x:x)\
        .zipWithIndex()\
        .map(lambda x:(x[0],x[1]))\
        .collectAsMap() #get dict like user_id:index

    c_dict_reverse = {j:i for (i,j) in c_dict.items()} # get dict like index:user_id

    re_num = min(round(len(b_dict)/1000),100)

    # (business_id,userid,score) --> (business,user,score)
    # -->(business,(user,score))
    # -->(business,{user:score...}) only consider those business have more than 50 ratings
    review_rdd = input_rdd.map(lambda x: (b_dict[x[1]],(c_dict[x[0]],float(x[2]))))\
        .groupByKey()\
        .filter(lambda x: len(x[1]) >= 40)\
        .repartition(re_num)\
        .mapValues(lambda x:dict(x))

    # business average rate:
    b_rate_dict = input_rdd.map(lambda x:(b_dict[x[1]],(float(x[2]),1)))\
        .reduceByKey(lambda x,y:(x[0]+y[0],x[1]+y[1]))\
        .mapValues(lambda x:x[0]/x[1])\
        .collectAsMap()

    # --> (user,(business,score)) --> (user,{business:score})
    user_rdd = input_rdd.map(lambda x: (c_dict[x[0]],(b_dict[x[1]],float(x[2]))))\
        .groupByKey()\
        .repartition(re_num)\
        .mapValues(lambda x:dict(x))

    # -->((business1,scorelist1),(businesss2,socrelist2)...)
    # -->((business1,business2),score)
    # -->{business1:{business2:score2...}}
    # filter all of pairs in which the pearson rating is more than 0

    pearson_rdd = review_rdd.cartesian(review_rdd)\
        .filter(lambda x:x[0][0] < x[1][0])\
        .mapPartitions(yield_pearson_weight)\
        .flatMap(lambda x:[((x[0][0],x[0][1]),x[1]),((x[0][1],x[0][0]),x[1])])\
        .map(lambda x:(x[0][0],(x[0][1],x[1])))\
        .groupByKey()\
        .mapValues(lambda x:dict(x))

    # we add the last filter to decrease the rdd volumn
    # -->{(business1,business2),score}

    # (user_index,item_index) --> (user,item) --> (user,(item,user_score)) -->(item,(user,user_score))
    # -->(item,((user,user_score),item_relation))
    # -->((user,item),(user_socre,item_relation))
    # --> (user_index,item_index,item_relation)
    predict_rdd = output_rdd.map(lambda x:(c_dict.get(x[0],x[0]),b_dict.get(x[1],x[1])))\
        .leftOuterJoin(user_rdd)\
        .map(lambda x:(x[1][0],(x[0],x[1][1])))\
        .leftOuterJoin(pearson_rdd)\
        .map(lambda x:((x[1][0][0],x[0]),predict(x[0],x[1][0][1],x[1][1],b_rate_dict)))\
        .map(lambda x:(c_dict_reverse.get(x[0][0],x[0][0])
                                      ,b_dict_reverse.get(x[0][1],x[0][1])
                        ,str(x[1])))

    predict_data = predict_rdd.collect()

    with open(output_file,'w+') as f:
        f.write('business_id_1, business_id_2, similarity\n')
        for line in predict_data:
            f.write(','.join(line))
            f.write('\n')

    end = time.time()
    print(f'time:{end-begin}')
































