from paprotka.integration.cython import faster, example


def add(a, b):
    return a + faster.fib(b)


def bar_delegate():
    return example.bar(10)
