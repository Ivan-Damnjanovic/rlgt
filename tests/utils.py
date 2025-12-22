import numpy as np


def batchify(test_cases, batch_size=2, expand_dims=True):
    """
    Turn a set of test cases into batch test cases by prepending the batch size
    to the test case and stacking every numpy array in the test case along the first axis
    to match the batch size.
    """

    new_test_cases = []

    for test_case in test_cases:
        new_test_case = list(test_case)
        for i, item in enumerate(test_case):
            if not isinstance(item, (np.ndarray, list)):
                continue

            if expand_dims:
                new_test_case[i] = np.stack([item] * batch_size, axis=0)
            else:
                new_test_case[i] = np.concatenate([item] * batch_size, axis=0)

        if expand_dims:
            new_test_case.insert(0, batch_size)
        else:
            new_test_case[0] = batch_size

        new_test_cases.append(tuple(new_test_case))

    return new_test_cases


def merge(test_cases, expand_dims=True):
    """
    Merge a list of test cases to create a batch test case by prepending the batch size
    and stacking all of the numpy arrays in each case along the first axis.
    """

    res = []
    batch_size = len(test_cases)

    for i in range(len(test_cases[0])):
        items = [test_case[i] for test_case in test_cases]
        if isinstance(items[0], (np.ndarray, list)):
            if expand_dims:
                res.append(np.stack(items, axis=0))
            else:
                res.append(np.concatenate(items, axis=0))
        else:
            assert len(set(items)) == 1
            res.append(items[0])

    if expand_dims:
        res.insert(0, batch_size)
    else:
        res[0] = batch_size

    return tuple(res)
