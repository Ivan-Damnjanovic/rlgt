from typing import Callable


RandomActionMechanism = Callable[[bool], float]


def create_constant_random_action_mechanism(
    random_action_probability: float,
) -> RandomActionMechanism:
    def result(is_best_score_improved: bool) -> float:
        return random_action_probability

    return result


def create_multiplication_factor_random_action_mechanism(
    initial_random_action_probability: float,
    waiting_period: int,
    multiplication_factor: float,
    maximum_random_action_probability: float,
) -> RandomActionMechanism:
    counter = 0
    random_action_probability = initial_random_action_probability

    def result(is_best_score_improved: bool) -> float:
        nonlocal counter, random_action_probability

        if is_best_score_improved:
            counter = 0
            random_action_probability = initial_random_action_probability

        else:
            counter += 1
            if counter >= waiting_period:
                counter -= waiting_period
                random_action_probability *= multiplication_factor
                random_action_probability = min(
                    random_action_probability, maximum_random_action_probability
                )

        return random_action_probability

    return result
