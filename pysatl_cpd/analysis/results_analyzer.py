class CpdResultsAnalyzer:
    """Class for counting confusion matrix and other metrics on CPD results"""

    @staticmethod
    def count_confusion_matrix(
        predicted: list[int], actual: list[int], window: tuple[int, int] | None = None
    ) -> tuple[int, int, int, int]:
        """static method for counting confusion matrix for hypothesis of equality of change points on a window

        :param: predicted: first array or list of change points, determined as prediction
        :param: actual: second array or list of change points, determined as actual
        :param: window: tuple of two indices (start, stop), determines a window for hypothesis

        :return: tuple of integers (true-positive, true-negative, false-positive, false-negative)
        """
        if not predicted and not actual:
            raise ValueError("no results and no predictions")
        if window is None:
            window = (min(predicted + actual), max(predicted + actual))
        predicted_set = set(predicted)
        actual_set = set(actual)
        tp = tn = fp = fn = 0
        for i in range(window[0], window[1]):
            if i in predicted_set:
                if i in actual_set:
                    tp += 1
                    continue
                fp += 1
            elif i in actual_set:
                fn += 1
                continue
            tn += 1
        return tp, tn, fp, fn

    @staticmethod
    def count_accuracy(predicted: list[int], actual: list[int], window: tuple[int, int] | None = None) -> float:
        """static method for counting accuracy metric for hypothesis of equality of change points on a window

        :param: predicted: first array or list of change points, determined as prediction
        :param: actual: second array or list of change points, determined as actual
        :param: window: tuple of two indices (start, stop), determines a window for hypothesis

        :return: float, accuracy metric
        """
        tp, tn, fp, fn = CpdResultsAnalyzer.count_confusion_matrix(predicted, actual, window)
        if tp + tn == 0:
            return 0.0
        return (tp + tn) / (tp + tn + fp + fn)

    @staticmethod
    def count_precision(predicted: list[int], actual: list[int], window: tuple[int, int] | None = None) -> float:
        """static method for counting precision metric for hypothesis of equality of change points on a window

        :param: predicted: first array or list of change points, determined as prediction
        :param: actual: second array or list of change points, determined as actual
        :param: window: tuple of two indices (start, stop), determines a window for hypothesis

        :return: float, precision metric
        """
        tp, _, fp, _ = CpdResultsAnalyzer.count_confusion_matrix(predicted, actual, window)
        if tp == 0:
            return 0.0
        return tp / (tp + fp)

    @staticmethod
    def count_recall(predicted: list[int], actual: list[int], window: tuple[int, int] | None = None) -> float:
        """static method for counting recall metric for hypothesis of equality of change points on a window

        :param: predicted: first array or list of change points, determined as prediction
        :param: actual: second array or list of change points, determined as actual
        :param: window: tuple of two indices (start, stop), determines a window for hypothesis

        :return: float, recall metric
        """
        tp, _, _, fn = CpdResultsAnalyzer.count_confusion_matrix(predicted, actual, window)
        if tp == 0:
            return 0
        return tp / (tp + fn)
