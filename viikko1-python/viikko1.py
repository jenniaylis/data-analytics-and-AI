import random
print ("hello world")

x = random.randint(1, 100)
print("first random number is ",str(x))

y = random.randint(1, 100)
print("another random number is ",str(y))

if(x>y):
    print(x, "is bigger in value than ", y)

if(y>x):
    print(y, "is bigger in value than", x)

if(y==x):
    print(y, "and", x, "are equal in value")


x = random.randint(1, 10)
y = random.randint(1, 10)

def summa(a, b):
    return a + b

print(x, "plus", y, "equals", summa(x, y))
