class CpdResultsAnalyzer:
    """Class for counting confusion matrix and other metrics on CPD results"""

    @staticmethod
    def count_confusion_matrix(
        result1: list[int], result2: list[int], window: tuple[int, int] | None = None
    ) -> tuple[int, int, int, int]:
        """static method for counting confusion matrix for hypothesis of equality of change points on a window

        :param: result1: first array or list of change points, determined as prediction
        :param: result2: second array or list of change points, determined as actual
        :param: window: tuple of two indices (start, stop), determines a window for hypothesis

        :return: tuple of integers (true-positive, true-negative, false-positive, false-negative)
        """
        if not result1 and not result2:
            raise ValueError("no results and no predictions")
        if window is None:
            window = (min(result1 + result2), max(result1 + result2))
        result1_set = set(result1)
        result2_set = set(result2)
        tp = tn = fp = fn = 0
        for i in range(window[0], window[1]):
            if i in result1_set:
                if i in result2_set:
                    tp += 1
                    continue
                fp += 1
            elif i in result2_set:
                fn += 1
                continue
            tn += 1
        return tp, tn, fp, fn

    @staticmethod
    def count_accuracy(result1: list[int], result2: list[int], window: tuple[int, int] | None = None) -> float:
        """static method for counting accuracy metric for hypothesis of equality of change points on a window

        :param: result1: first array or list of change points, determined as prediction
        :param: result2: second array or list of change points, determined as actual
        :param: window: tuple of two indices (start, stop), determines a window for hypothesis

        :return: float, accuracy metric
        """
        tp, tn, fp, fn = CpdResultsAnalyzer.count_confusion_matrix(result1, result2, window)
        if tp + tn == 0:
            return 0.0
        return (tp + tn) / (tp + tn + fp + fn)

    @staticmethod
    def count_precision(result1: list[int], result2: list[int], window: tuple[int, int] | None = None) -> float:
        """static method for counting precision metric for hypothesis of equality of change points on a window

        :param: result1: first array or list of change points, determined as prediction
        :param: result2: second array or list of change points, determined as actual
        :param: window: tuple of two indices (start, stop), determines a window for hypothesis

        :return: float, precision metric
        """
        tp, tn, fp, fn = CpdResultsAnalyzer.count_confusion_matrix(result1, result2, window)
        if tp == 0:
            return 0.0
        return tp / (tp + fp)

    @staticmethod
    def count_recall(result1: list[int], result2: list[int], window: tuple[int, int] | None = None) -> float:
        """static method for counting recall metric for hypothesis of equality of change points on a window

        :param: result1: first array or list of change points, determined as prediction
        :param: result2: second array or list of change points, determined as actual
        :param: window: tuple of two indices (start, stop), determines a window for hypothesis

        :return: float, recall metric
        """
        tp, tn, fp, fn = CpdResultsAnalyzer.count_confusion_matrix(result1, result2, window)
        if tp == 0:
            return 0
        return tp / (tp + fn)
