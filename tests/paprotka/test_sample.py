from paprotka import sample


def should_add_two_numbers():
    assert sample.add(10, 10) == 65
