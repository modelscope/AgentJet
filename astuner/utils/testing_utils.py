from astuner.utils.dynamic_import import dynamic_import


def get_test_lambda(test_name):
    test_cls = dynamic_import(test_name)()
    print("using test lambda:", test_name)
    return test_cls


def _test_if_test_mode(key, value, config):
    if not config.astuner.execute_test:
        return
    if config.astuner.execute_test == "do_not_test":
        return
    test_lambda = get_test_lambda(config.astuner.execute_testing_lambda)
    return test_lambda(key, value)


def _mock_if_test_mode(key, value, config):
    if not config.astuner.execute_test:
        return value
    if config.astuner.execute_test == "do_not_test":
        return value
    test_lambda = get_test_lambda(config.astuner.execute_testing_lambda)
    return test_lambda.mock(key)


class GoodbyeException(Exception):
    pass


class TestFailException(Exception):
    pass
