from pyspark import SparkContext
import json
import time
import operator
from collections import OrderedDict
import sys

# input_file = './test_review.json'
# output_file = './output2.json'
# n = 3
if __name__ == '__main__':
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    n = sys.argv[3]
    sc = SparkContext.getOrCreate()
    result = OrderedDict()

    # automatic
    rdd = sc.textFile(input_file).map(lambda x: json.loads(x)).cache()
    start = time.time()
    rdd1 = rdd.map((lambda x: (x['business_id'], 1)))
    answer = rdd1.reduceByKey(operator.add).takeOrdered(10, key=lambda x: (-x[1], x[0]))
    end = time.time()
    exe_time1 = end - start
    n_partition1 = rdd1.getNumPartitions()
    n_items1 = rdd1.glom().map(len).collect()
    default_dict = OrderedDict()
    default_dict["n_partition"] = n_partition1
    default_dict["n_items"] = n_items1
    default_dict["exe_time"] = exe_time1
    result["default"] = default_dict

    # customized
    start = time.time()
    rdd2 = rdd.map(lambda x: (x['business_id'], 1)).partitionBy(n, lambda x: ord(x[0]) - ord(x[-1]))
    answer = rdd2.reduceByKey(operator.add).takeOrdered(10, key=lambda x: (-x[1], x[0]))

    end = time.time()
    exe_time2 = end - start
    n_partition2 = rdd2.getNumPartitions()
    n_items2 = rdd2.glom().map(len).collect()
    result["customized"] = {
        "n_partition": n_partition2,
        "n_items": n_items2,
        "exe_time": exe_time2
    }
    customized_dict = OrderedDict()
    customized_dict["n_partition"] = n_partition2
    customized_dict["n_items"] = n_items2
    customized_dict["exe_time"] = exe_time2
    result["customized"] = customized_dict

    with open(output_file, 'w') as f:
        json.dump(result, f)
    f.close()