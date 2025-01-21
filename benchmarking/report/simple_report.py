from pathlib import Path
from dataclasses import dataclass


# User can filter out none values and use only some.
# Actually, Measures can be class with method to incalsulate this work.
class Measures():
    def __init__(self) -> None:
        time: float | None
        memory: float | None # TODO: There are different types of memory
        power: float | None
        f1: float | None
        sl: float | None
        interval: int | None

    def filterOutNone(self) -> dict[str, float]:
        ...


class BenchmarkingReport():
    def __init__(self, resultsDir: Path) -> None:
        self.__resultsDir = resultsDir
        self.__result: Measures = Measures()

    def countTime(self) -> None:
        ...

    def countMemory(self) -> None:
        ...

    def countPower(self) -> None:
        ...

    def countF1(self) -> None:
        ...

    def countSL(self) -> None:
        ...

    def countInterval(self) -> None:
        ...

    # Either serialized format or just string. But probably we need to count some statistics over results, so it would be good to make it serialized to deserialized and get some metrics.
    # We do not need to serialize it. We can just return some dataclass containig. Products do not need to have general interface. What if we do not need some metrics?
    # Make fields optional? Yeah.
    def getResult(self) -> str:
        ...