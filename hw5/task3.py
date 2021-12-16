from blackbox import BlackBox
import random
import sys

if __name__ == '__main__':
    file_name = sys.argv[1]
    stream_size = int(sys.argv[2])
    total_time = int(sys.argv[3])  # number of ask
    output_file = sys.argv[4]

    # file_name = 'users.txt'
    # stream_size = 100
    # total_time = 30  # number of ask
    # output_file = 'test3.csv'

    bx = BlackBox()
    random.seed(553)
    stream_users = bx.ask(file_name, stream_size)
    sample = stream_users
    n = len(sample)
    s = len(sample)
    output = [[100,sample[0],sample[20],sample[40],sample[60],sample[80]]]
    for i in range(total_time-1): # only need to generate 29times since the output of first time is given
        stream_users = bx.ask(file_name, stream_size)
        for user in stream_users:
            n+=1
            random_num = random.random()
            if random_num < s/n:
                replace_index = random.randint(0,s-1)
                sample[replace_index] = user
        output.append([100*(i+2),sample[0],sample[20],sample[40],sample[60],sample[80]])

    with open(output_file, 'w+') as f:
        f.write('seqnum,0_id,20_id,40_id,60_id,80_id\n')
        for i in output:
            string_list = [str(value) for value in i]
            f.write(','.join(string_list))
            f.write('\n')

