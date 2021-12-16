from pyspark import SparkContext, SparkConf
import json
import operator
import sys

# input_file = 'test_review.json'
# output_file = 'output.json'

if __name__ == '__main__':
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    sc = SparkContext("local","task1").getOrCreate()
    rddfile = sc.textFile(input_file).map(lambda x: json.loads(x))

    # A. The total number of reviews
    n_review = rddfile.map(lambda x:(x,1)).count()

    # B. The number of reviews in 2018
    n_review_2018 = rddfile.filter(lambda x:x['date'][:4] == '2018').count()

    # C. The number of distinct users who wrote reviews
    n_user = rddfile.filter(lambda x:x['text'] is not None).map(lambda x:x['review_id'])\
        .distinct().count()

    # D. The top 10 users who wrote the largest numbers of reviews and the number of reviews they wrote
    n = rddfile.map(lambda x: (x["user_id"],1)).reduceByKey(operator.add).takeOrdered(10,key=lambda x:(-x[1],x[0]))

    # E. The number of distinct businesses that have been reviewd
    n_business = rddfile.map(lambda x: (x['business_id'],1)).distinct().count()

    # F. The top10 businesses that had the largest numbers of reviews and the number of reviews they had
    m = rddfile.map(lambda x:(x['business_id'],1))\
        .reduceByKey(operator.add).takeOrdered(10,key=lambda x:(-x[1],x[0]))

    final = dict()
    final['n_review'] = n_review
    final['n_review_2018'] = n_review_2018
    final['n_user'] = n_user
    final['top10_user'] = n
    final['n_business'] = n_business
    final['top10_business'] = m

    with open(output_file, 'w') as f:
        json.dump(final,f)
    f.close()
