def calc(n):
    return (3 * n) + 1


x = 1
while True:
    n = calc(x)
    print(n)
    x = n
    i = input('Press any key to continue')