from paprotka.integration import cython_example


def should_add_two_numbers():
    assert cython_example.add(10, 10) == 65
