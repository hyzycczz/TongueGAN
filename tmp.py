def three(tmp):
    return 1,2,3

a, b, c = [three(block) for block in range(5)]
print(a)
print(b)
print(c)
