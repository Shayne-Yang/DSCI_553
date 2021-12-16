import time
from pyspark import SparkContext
import json
import sys

if __name__ == '__main__':
    # review_file = 'test_review.json'
    # business_file = 'business.json'
    # output_file1 = 'output3_1.txt'
    # output_file2 = 'output3_2.json'
    review_file = sys.argv[1]
    business_file = sys.argv[2]
    output_file1 = sys.argv[3]
    output_file2 = sys.argv[4]

    sc = SparkContext.getOrCreate()

    # 3A

    def replace_none(x):
        if x[0] is None:
            x[0] = ""
        return x

    business_rdd = sc.textFile(business_file).map(lambda x: json.loads(x))\
        .map(lambda x:(x['business_id'],x['city']))

    review_rdd = sc.textFile(review_file).map(lambda x: json.loads(x))\
        .map(lambda x:(x['business_id'],x['stars']))\
        .filter(lambda x:x[1] is not None)\
        .groupByKey()\
        .map(lambda x:(x[0],(sum(x[1]),len(x[1]))))

    # join the dataframe
    all = business_rdd.leftOuterJoin(review_rdd)\
        .map(lambda x: replace_none(x[1]))\
        .map(lambda x:(x[0],x[1]))\
        .filter(lambda x:x[1] is not None)\
        .reduceByKey(lambda  a,b: (a[0]+b[0],a[1]+b[1]))\
        .mapValues(lambda x:float(x[0]/x[1]))\
        .collect()

    final = sorted(all,key = lambda x: (-x[1],x[0]))

    with open(output_file1, 'w+') as a:
        a.write('city,stars'+'\r')
        for i in range(len(final)):
            line = str(final[i][0] + ',' +str(final[i][1]))
            a.write(line + '\r')

    # 3B
    # Method1
    start = time.clock()
    def replace_none(x):
        if x[0] is None:
            x[0] = ""
        return x

    business_rdd = sc.textFile(business_file).map(lambda x: json.loads(x))\
        .map(lambda x:(x['business_id'],x['city']))

    review_rdd = sc.textFile(review_file).map(lambda x: json.loads(x))\
        .map(lambda x:(x['business_id'],x['stars']))\
        .filter(lambda x:x[1] is not None)\
        .groupByKey()\
        .map(lambda x:(x[0],(sum(x[1]),len(x[1]))))

    # join the dataframe
    all = business_rdd.leftOuterJoin(review_rdd)\
        .map(lambda x: replace_none(x[1]))\
        .map(lambda x:(x[0],x[1]))\
        .filter(lambda x:x[1] is not None)\
        .reduceByKey(lambda  a,b: (a[0]+b[0],a[1]+b[1]))\
        .mapValues(lambda x:float(x[0]/x[1]))\
        .collect()

    final = sorted(all,key = lambda x: (-x[1],x[0]))
    print(final[:10])
    end = time.clock()
    m1 = end - start

    # Method2

    start = time.clock()
    def replace_none(x):
        if x[0] is None:
            x[0] = ""
        return x

    business_rdd = sc.textFile(business_file).map(lambda x: json.loads(x))\
        .map(lambda x:(x['business_id'],x['city']))

    review_rdd = sc.textFile(review_file).map(lambda x: json.loads(x))\
        .map(lambda x:(x['business_id'],x['stars']))\
        .filter(lambda x:x[1] is not None)\
        .groupByKey()\
        .map(lambda x:(x[0],(sum(x[1]),len(x[1]))))

    # join the dataframe
    all = business_rdd.leftOuterJoin(review_rdd)\
        .map(lambda x: replace_none(x[1]))\
        .map(lambda x:(x[0],x[1]))\
        .filter(lambda x:x[1] is not None)\
        .reduceByKey(lambda  a,b: (a[0]+b[0],a[1]+b[1]))\
        .mapValues(lambda x:float(x[0]/x[1]))\
        .takeOrdered(10,key = lambda x: (-x[1],x[0]))

    print(all)
    end = time.clock()
    m2 = end - start

    output2 = dict()
    output2['m1'] = m1
    output2['m2'] = m2
    output2['reason'] = ''

    print(output2)

    with open(output_file2, 'w') as a:
        json.dump(output2, a)
    a.close()