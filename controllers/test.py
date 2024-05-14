arr = [i for i in range(640)]
a = 22
for i in range(3,10000000):
    b = arr[::i]
    if (b[-1] == 638):
        print(i)
        break
print(b)
print(len(b))