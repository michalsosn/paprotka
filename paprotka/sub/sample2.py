import paprotka.sub.faster as faster


def foo(a, b):
    return faster.fib(a) + faster.fib(b)


def bar(a):
    return a * a
