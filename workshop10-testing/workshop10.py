'''
Author: Akond Rahman 
Code needed for Workshop 10
'''

import random
import json
import traceback

def divide(v1, v2):
    # Error 1: Division by zero error
    if v2 == 0:
        raise ValueError("Division by zero error")
    return v1 / v2

def fuzzValues(val1, val2):
    # Error 2: Invalid input types
    if not isinstance(val1, (int, float)) or not isinstance(val2, (int, float)):
        raise TypeError("Invalid input types")

    # Error 3: Overflow or underflow errors
    if abs(val1) == float('inf') or abs(val2) == float('inf'):
        raise OverflowError("Overflow or underflow error")

    # Error 4: Loss of precision errors
    res = divide(val1, val2)
    if isinstance(res, float) and round(res, 5) != res:
        raise ArithmeticError("Loss of precision error")
    return res

def simpleFuzzer():
    test_values = [
        # Error 1: Division by zero error
        (10, 0),

        # Error 2: Invalid input types
        ("string", 5),  # First argument is a string

        # Error 3: Overflow or underflow errors
        (float('inf'), 2),  # First argument is infinity

        # Error 4: Loss of precision errors
        (0.1, 3),  # Values that might cause precision loss

        # Error 5: Invalid operation
        (10, "string"),  # Second argument is a string

        # Error 6: Unhandled exception
        (None, 5),  # First argument is None

        # Error 7: Unexpected behavior due to rounding errors
        (1, 3)  # Values that might cause rounding errors
    ]

    for val1, val2 in test_values:
        try:
            fuzzValues(val1, val2)
        except Exception as exc:
            traceback.print_exc()


if __name__=='__main__':
    simpleFuzzer()
