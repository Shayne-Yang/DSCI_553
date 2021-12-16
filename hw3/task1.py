from pyspark import SparkContext, SparkConf
import random
import sys
import time
from itertools import combinations


# get the dummy rdd

def dummy(list, k):
    dummy_line = [0 for n in range(k)]
    for a in list:
        dummy_line[a] = 1
    return dummy_line

def Hashfunc_list(bins, num=100):
    func_list = list()

    param_as = random.sample(range(1, sys.maxsize - 1), num)
    param_bs = random.sample(range(0, sys.maxsize - 1), num)

    def build_func(a, b, m):
        def hash_funcs(x):
            return ((a * x + b) % 233333333333) % m

        return hash_funcs

    for a, b in zip(param_as, param_bs):
        func_list.append(build_func(a, b, bins))

    return func_list

# minhash to generate signiture

def minhash(list, hash_list):
    inf = len(list) + 1  # generate a very large integer
    hash_v = [inf for i in range(len(hash_list))]  # initialize the minhash signiture
    for i in range(len(list)):  # for row i
        if list[i] == 1:
            for j in range(len(hash_list)):  # for each hash function
                if hash_v[j] > hash_list[j](i):
                    hash_v[j] = hash_list[j](i)
    return hash_v



def jaccard_similarity(s1, s2):
    denominator = len(s1|s2)
    nominator = len(s1&s2)
    return nominator / denominator


if __name__ == '__main__':
    input_file = 'yelp_train.csv'
    output_file = 'test.csv'
    # input_file = sys.argv[1]
    # output_file = sys.argv[2]

    # spark settings
    conf = SparkConf().setMaster("local") \
        .setAppName("hw_3_task1") \
        .set("spark.executor.memory", "15g") \
        .set('spark.executor.cores', '10') \
        .set("spark.driver.memory", "20g") \
        .set('spark.executor.instances', '20')

    sc = SparkContext(conf=conf)
    sc.setLogLevel("WARN")
    start = time.time()

    # band,row number, and hash function we need
    b = 100
    r = 2
    hash_num = b * r

    # read the file
    rdd = sc.textFile(input_file) \
        .map(lambda x: x.split(','))
    header = rdd.first()
    rdd = rdd.filter(lambda x: x != header) \
        .map(lambda x: (x[1], x[0])).cache()  # skip the header
    #I mistake the question by trying to solve the similar user rather than similar business.
    #I changed the index above but do not change the alias below

    # get the business, customer dictionary and get their index to minimize the memory utility
    c_dict = rdd.map(lambda x: x[0]) \
        .distinct() \
        .sortBy(lambda x: x) \
        .zipWithIndex() \
        .map(lambda x: (x[0], x[1])) \
        .collectAsMap()

    c_len = len(c_dict)

    c_dict_reverse = rdd.map(lambda x: x[0]) \
        .distinct() \
        .sortBy(lambda x: x) \
        .zipWithIndex() \
        .map(lambda x: (x[1], x[0])) \
        .collectAsMap()

    bu_dict = rdd.map(lambda x: x[1]) \
        .distinct() \
        .zipWithIndex() \
        .map(lambda x: (x[0], x[1])) \
        .collectAsMap()

    bu_len = len(bu_dict)  # get the total number of distinct business

    # get one-hot rdd
    cb_summary = rdd.map(lambda x: (c_dict[x[0]], bu_dict[x[1]])) \
        .groupByKey()\
        .repartition(min(round(c_len/1000),100)) \
        .cache() # the partition numb cannot more than the cores number
         # repartition to increase the speed

    one_hot_rdd = cb_summary.mapValues(lambda x: dummy(x, bu_len))

    # get minhash rdd

    hash_list = Hashfunc_list(bins=bu_len * 2,
                              num = hash_num)  # suppose the number of buckets should be at least larger than the number
    # of business

    minhash_rdd = one_hot_rdd.mapValues(lambda x: minhash(x, hash_list))

    candidate_rdd = minhash_rdd \
        .flatMap(lambda x: [(tuple(x[1][r * i:r * (i + 1)] + [i]), x[0]) for i in range(b)]) \
        .groupByKey() \
        .flatMap(lambda x: [k for k in list(combinations(sorted(x[1]), 2))]) \
        .distinct() \
        .collect()

    # Use this similarity to calculate the real similarity

    pair_candidate = sorted(candidate_rdd, key=(lambda x: (x[0],x[1])))
    candidate_list = set([i for tuple in candidate_rdd for i in tuple])  # get all of distinct candidate

    # final similarity
    # similarity_rdd = one_hot_rdd.filter(lambda x: x[0] in candidate_list).collectAsMap()
    cb_dictionay = cb_summary\
        .filter(lambda x:x[0] in candidate_list)\
        .map(lambda x:(x[0],set(x[1])))\
        .collectAsMap()

    # print('rdd finish')
    with open(output_file,'w+') as f:
        f.write('business_id_1, business_id_2, similarity\n')
        for pair in pair_candidate:
            similar = jaccard_similarity(cb_dictionay[pair[0]], cb_dictionay[pair[1]])
            if similar >= 0.5:
                f.write(','.join([c_dict_reverse[pair[0]], c_dict_reverse[pair[1]], str(similar)]))
                f.write('\n')

    end = time.time()
    # print(f'execute time:{end - start}')
