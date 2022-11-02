import random

def multiplication(a, b):
    return a * b

print("Calculate following exercises:")

list = []
correct = 0
count = 0
while count < 5:
    a = random.randint(1, 10)
    b = random.randint(1, 10)
    
    answer = int(input(f'{a} * {b} = '))

    if (answer == multiplication(a, b)):
        correct = correct + 1
        list.append(f'Correct :-) {a} * {b} = {answer}')
    else:
        list.append(f'False :-( Correct answer is {a} * {b} = {multiplication(a, b)}')

    count = count + 1

for x in list:
    print(x)
print(f"you got {correct}/5 points!")