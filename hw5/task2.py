from blackbox import BlackBox
import random
import sys
import binascii
from statistics import median

# generate hash functions based on how many you want
def hash_fun(num=3):
    func_list = []

    random.seed(0)
    param_as = random.sample(range(1, sys.maxsize - 1), num)
    param_bs = random.sample(range(0, sys.maxsize - 1), num)
    random.seed(None)

    def build_func(a, b, m=69997):
        def hash_funcs(x):
            return ((a * x + b) % 233333333333) % m

        return hash_funcs

    for a, b in zip(param_as, param_bs):
        func_list.append(build_func(a, b))

    return func_list

# convert a user_id string to int
def hexlify(s):
    return int(binascii.hexlify(s.encode('utf8')), 16)


def myhashs(s):
    result = []
    hash_fun_list = hash_fun(200)  # suppose there're 200 hash_fun
    for f in hash_fun_list:
        result.append(f(hexlify(s)))
    return result

# return the number of trailing zero
def trailing_zeros(int):
    s = str(bin(int)[2:])
    return len(s) - len(s.rstrip('0'))

if __name__ == '__main__':
    file_name = sys.argv[1]
    stream_size = int(sys.argv[2])
    total_time = int(sys.argv[3])  # number of ask
    output_file = sys.argv[4]

    # file_name = 'users.txt'
    # stream_size = 300
    # total_time = 30  # number of ask
    # output_file = 'test2.csv'

    bx = BlackBox()
    output = []

    for i in range(total_time):
        stream_users = bx.ask(file_name, stream_size)
        max_trailing_zeros = [0 for i in range(500)]
        for user in stream_users:
            hash_value = myhashs(user)
            trailing_zero = [trailing_zeros(value) for value in hash_value]
            update = [max(max_trailing_zeros[i],trailing_zero[i]) for i in range(len(trailing_zero))]
            max_trailing_zeros = update
        estimate_number = int(median(2**i for i in max_trailing_zeros)) #use median to batch estimate
        real_number = len(set(stream_users))
        output.append([i,real_number,estimate_number])

    with open(output_file,'w+') as f:
        f.write('Time,Ground Truth,Estimation\n')
        for i in output:
            string_list = [str(value) for value in i]
            f.write(','.join(string_list))
            f.write('\n')