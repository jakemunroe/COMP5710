def performAdd(c, d):
    return c + d 


def performSub(x, y):
    return x - y 

def performMult(a, b):
    return a * b

def performDiv(a, b):
    try:
        return a / b
    except ZeroDivisionError as e:
        return "Error: Cannot Divide by Zero"