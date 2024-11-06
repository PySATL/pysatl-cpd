import math

from CPDShell.Core.algorithms.GpraphCPD.abstracts.igraph import IGraph
from CPDShell.Core.algorithms.GpraphCPD.abstracts.igraph_cpd import IGraphCPD


class GraphCPD(IGraphCPD):
    def __init__(self, graph: IGraph):
        super().__init__(graph)

    def calculation_e(self, thao: int) -> float:
        p1 = ((2 * thao) * (self.size - thao)) / (self.size * (self.size - 1))
        return p1 * self.graph.num_of_edges

    def calculation_var(self, thao: int) -> float:
        p1 = ((2 * thao) * (self.size - thao)) / (self.size * (self.size - 1))
        p2 = (4 * thao * (thao - 1) * (self.size - thao) * (self.size - thao - 1)) / (
            self.size * (self.size - 1) * (self.size - 2) * (self.size - 3)
        )
        var = (
            p1 * self.graph.num_of_edges
            + (0.5 * p1 - p2) * self.graph.sum_of_squares_of_degrees_of_nodes()
            + (p2 - p1**2) * self.graph.num_of_edges**2
        )
        return var

    def calculation_z(self, thao: int) -> float:
        zg = -((self.graph.check_edges_exist(thao) - self.calculation_e(thao)) / math.sqrt(self.calculation_var(thao)))
        return zg

    def calculation_probability(self, lower_order: int, higher_order: int) -> float:
        n = higher_order - lower_order
        combinations = (n * (n - 1) * (n - 2)) / 6
        p = (self.graph.sum_triangle(lower_order, higher_order) / combinations) ** (1 / 3)
        return p

    def probability_diff(self):
        change_point = []
        probability = None
        threshold = 0.25
        window_zise = 20

        for thao in range(0, self.size, window_zise):
            current_probability = self.calculation_probability(thao, thao + window_zise)
            print(f"thao = {thao}, current_probability = {current_probability}, previous_probability = {probability}")

            if probability is None and current_probability != 0.0:
                probability = current_probability
            elif abs(current_probability - probability) > threshold:
                change_point.append(thao)
                probability = current_probability
            else:
                probability = current_probability

        return change_point

    def find_changepoint(self, border: float) -> list[int]:
        change_point_list: list[int] = []
        for t in range(1, self.size):
            if self.calculation_z(t) > border:
                change_point_list.append(t)
        return change_point_list
