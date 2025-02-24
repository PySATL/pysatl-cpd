from dataclasses import dataclass


@dataclass
class CpdProblem:
    """Specification of the solving problem

    :param to_localize: is it necessary to localize change points, defaults to False
    """

    to_localize: bool = True
