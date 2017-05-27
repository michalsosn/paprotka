class Union:
    def __init__(self, value=None):
        self.root = None
        self.value = value

    def getv(self):
        return self.resolve().value

    def setv(self, value):
        self.resolve().value = value

    def merge(self, other, merger=None):
        first_root = self.resolve()
        second_root = other.resolve()
        if first_root != second_root:
            if merger is not None:
                first_root.value = merger(first_root.value, second_root.value)
            second_root.value = None
            second_root.root = first_root

    def same(self, other):
        return self.resolve() == other.resolve()

    def resolve(self):
        if self.root is not None:
            self.root = self.root.resolve()
            return self.root
        return self
