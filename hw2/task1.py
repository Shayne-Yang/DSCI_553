from pyspark import SparkContext
import math
import time
import collections
import sys
import operator
from itertools import combinations


## for the single item
# find candidate with one item
def find_candidate(basket, sub_support, previous_out=None):
    counting = {}
    # Counting the number of all items
    for list in basket:
        for item in list:
            if item not in counting.keys():
                counting[item] = 1
            else:
                counting[item] += 1

    for item, num in counting.items():
        if num >= sub_support:
            yield (item, 1)


# find candidate with more than one item
def find_candidate2(basket, sub_support, previous_op):
    counting = {key: 0 for key in previous_op}
    for list in basket:
        for item in previous_op:
            if all(a in list for a in item):
                counting[item] += 1
            if counting[item] >= sub_support:
                previous_op.remove(item)
                yield (item, 1)


def find_final(basket, candidate):
    for list in basket:
        for item in candidate:
            # if the each candidate is a singleton
            if type(item) == type('a'):
                if item in list:
                    yield (item, 1)
                # if the each candidate has more than one element
            else:
                # if all of elements in one candidates are in the one list
                if all(k in list for k in item):
                    yield (item, 1)


def generate_next_candidate(single_item, previous_candidate):
    n = len(previous_candidate[0]) + 1
    return [triple for triple in combinations(candidate_single_rdd, n) if
            all(pair in pair_rdd for pair in combinations(triple, n - 1))]


def dedupe(items):
    seen = set()
    for tuple in items:
        for item in tuple:
            if item not in seen:
                yield item
                seen.add(item)


if __name__ == '__main__':
    start = time.time()
    case_number = int(sys.argv[1])  # 1 for Case 1 and 2 for Case 2
    support = int(sys.argv[2])
    input_file = sys.argv[3]
    output_file = sys.argv[4]
    # Support: integer that defines the minimum count to qualify as a frequent itemset
    # support = 9
    # case_number = 2
    # input_file = './small2.csv'
    # output_file = './output.csv'
    m = math.floor(math.sqrt(support))
    sub_support = math.floor(support / m)

    sc = SparkContext("local", "task1").getOrCreate()

    # initialize the partition number as 3 to debug

    sort_key = lambda x: (len(x), x)
    if case_number == 1:
        user_basket = sc.textFile(input_file, m) \
            .map(lambda line: line.split(",")) \
            .filter(lambda line: len(line) > 1) \
            .map(lambda line: (line[0], line[1])) \
            .groupByKey() \
            .map(lambda user_items: (user_items[0], sorted(list(set(list(user_items[1]))), key=lambda x: (len(x), x)))) \
            .map(lambda item_users: item_users[1])
    if case_number == 2:
        user_basket = sc.textFile(input_file, m) \
            .map(lambda line: line.split(",")) \
            .filter(lambda line: len(line) > 1) \
            .map(lambda line: (line[1], line[0])) \
            .groupByKey() \
            .map(lambda user_items: (user_items[0], sorted(list(set(list(user_items[1]))), key=lambda x: (len(x), x)))) \
            .map(lambda item_users: item_users[1])

    # print(f'the number of partition is {m}')
    # single candidate
    candidate_single_rdd = \
        user_basket.mapPartitions(lambda partition: find_candidate(basket=partition,
                                                                   sub_support=sub_support)) \
            .reduceByKey(lambda a, b: min(a, b)) \
            .sortByKey() \
            .map(lambda x: (x[0])) \
            .collect()
    candidate_collection = [sorted(candidate_single_rdd)]
    # print(candidate_single_rdd)
    # second pass
    single_rdd = \
        user_basket.mapPartitions(lambda partition: find_final(basket=partition,
                                                               candidate=sorted(candidate_single_rdd))) \
            .reduceByKey(lambda a, b: a + b) \
            .filter(lambda x: x[1] >= support) \
            .map(lambda x: x[0]) \
            .collect()
    # print(sorted(single_rdd))
    frequent_collection = [sorted(single_rdd)]
    # for the following turn
    previous = [(a, b) for index_a, a in enumerate(sorted(single_rdd)) for index_b, b in
                enumerate(sorted(single_rdd)) if index_b > index_a]
    # item number in each basket
    num = 2
    while previous:
        # first pass
        pair_candidate_rdd = user_basket.mapPartitions(lambda partition: find_candidate2(basket=partition,
                                                                                         sub_support=sub_support,
                                                                                         previous_op=previous)) \
            .reduceByKey(lambda a, b: min(a, b)) \
            .sortByKey() \
            .map(lambda x: (x[0])) \
            .collect()
        # print(sorted(pair_candidate_rdd))
        # second pass
        candidate_collection.append(sorted(pair_candidate_rdd, key=sort_key))
        pair_rdd = user_basket.mapPartitions(lambda partition: find_final(basket=partition,
                                                                          candidate=pair_candidate_rdd)) \
            .reduceByKey(lambda a, b: a + b) \
            .filter(lambda x: x[1] >= support) \
            .map(lambda x: (x[0])) \
            .collect()
        # print(sorted(pair_rdd))
        frequent_collection.append(sorted(pair_rdd, key=sort_key))
        num += 1
        single_set = sorted(dedupe(pair_rdd),key=sort_key)
        previous = sorted([triple for triple in combinations(single_set, num) if
                           all(pair in pair_rdd for pair in combinations(triple, num - 1))], key=sort_key)
    # print(candidate_collection)
    # print(frequent_collection)
    # output file
    with open(output_file, 'w+') as f:
        f.write('Candidates\n')
        f.write("('")
        f.write("'),('".join(candidate_collection[0]))
        f.write("')\n\n")
        for item in candidate_collection[1:]:
            f.write(','.join(map(str, item)))
            f.write('\n\n')
        f.write('Frequent Itemsets\n')
        f.write("('")
        f.write("'),('".join(frequent_collection[0]))
        f.write("')\n\n")
        for item in frequent_collection[1:]:
            f.write(','.join(map(str, item)))
            f.write('\n\n')

    end = time.time()
    print(f'Duration: {end - start}')
