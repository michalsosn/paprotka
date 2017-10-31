from paprotka.integration.cython import faster


def foo(a, b):
    return faster.fib(a) + faster.fib(b)


def bar(a):
    return a * a
