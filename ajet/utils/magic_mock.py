class SpecialMagicMock(object):
    def __init__(self, allowed_attributes=[]):
        # Use __dict__ to avoid triggering __setattr__
        self.__dict__["allowed_attributes"] = allowed_attributes
        self.__dict__["attr_store"] = {}

    def __getattr__(self, name):
        if name in self.allowed_attributes:
            return self.attr_store.get(name)
        else:
            raise ValueError(f"Attribute {name} is not allowed.")

    def __setattr__(self, name, value):
        if name in self.allowed_attributes:
            # Use __dict__ to avoid recursion
            self.__dict__["attr_store"][name] = value
        elif name in ("allowed_attributes", "attr_store"):
            # Allow setting internal attributes directly
            self.__dict__[name] = value
        else:
            raise ValueError(f"Attribute {name} is not allowed.")
