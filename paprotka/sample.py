from paprotka.sub.faster import fib
from paprotka.sub import sample2


def add(a, b):
    return a + fib(b)


def bar_delegate():
    return sample2.bar(10)