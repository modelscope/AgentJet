_this_is_a_test_dictionary = {}


class ExampleClass(object):
    def __init__(self):
        print("cls::" + str(id(_this_is_a_test_dictionary)))
        self.value = id(_this_is_a_test_dictionary)

    def get_value(self):
        return self.value
