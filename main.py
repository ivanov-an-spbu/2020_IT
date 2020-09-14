# from fib_generator import fib

import random


# def get_missed_doubled(a):
#     missed = a[0]
#     doubled = a[7]


a = list(range(1, 10))

random.shuffle(a)
missed = a[0]
doubled = a[7]
a[0] = a[7]
random.shuffle(a)

print(a)
print(missed, doubled)


res_doubled = None
missed_doubled = None

for i in a:
    i = abs(i)
    if a[i-1]<0:
        res_doubled = i
    else:
        a[i-1]*=-1

for i in range(len(a)):
    if a[i]>0:
        missed_doubled = i+1
        break

print(a)
print(res_doubled, missed_doubled)




