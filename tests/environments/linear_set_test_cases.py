from .linear_build_test_cases import (
    TEST_CASES_CONSTRUCTOR as linear_build_constructor_test_cases,
    TEST_CASES_RESET_BATCH as linear_build_reset_batch_test_cases,
)

TEST_CASES_CONSTRUCTOR = [
    (*test_case[:-1], lambda *_: None, test_case[-1])
    for test_case in linear_build_constructor_test_cases
] + [(*test_case[:-1], None, test_case[-1]) for test_case in linear_build_constructor_test_cases]

TEST_CASES_RESET_BATCH = [test_case + (None,) for test_case in linear_build_reset_batch_test_cases]
