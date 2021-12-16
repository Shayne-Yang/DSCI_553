from blackbox import BlackBox
import random
import sys
import binascii


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
    hash_fun_list = hash_fun(16)# suppose there's three hash_fun
    for f in hash_fun_list:
        result.append(f(hexlify(s)))
    return result




if __name__ == '__main__':
    # file_name = 'users.txt'
    # stream_size = 100
    # total_time = 30  # number of ask
    # output_file = 'test1.csv'
    file_name = sys.argv[1]
    stream_size = int(sys.argv[2])
    total_time = int(sys.argv[3])  # number of ask
    output_file = sys.argv[4]

    bx = BlackBox()
    stream_users = bx.ask(file_name, stream_size)

    global_hash_vector = [0 for i in range(69997)]
    output = []
    seen = set()

    for i in range(total_time):
        stream_users = bx.ask(file_name,stream_size)
        hash_used_show = list()

        for user in stream_users:
            hash_value = myhashs(user)
            if sum(global_hash_vector[value] for value in hash_value) == len(hash_value):
                hash_used_show.append(user)

        true_used_show = [user for user in stream_users if user in seen]
        if len(hash_used_show) == 0:
            output.append([i,0.0])
        else:
            output.append([i,len(list(user for user in hash_used_show if user not in true_used_show))/len(hash_used_show)])

        # refresh the seen set and global hash vector
        for user in stream_users:
            seen.add(user)
            hash_value = myhashs(user)
            for value in hash_value:
                global_hash_vector[value] = 1

    print(output)

    with open(output_file,'w+') as f:
        f.write('Time,FPR\n')
        for i in output:
            string_list = [str(value) for value in i]
            f.write(','.join(string_list))
            f.write('\n')




