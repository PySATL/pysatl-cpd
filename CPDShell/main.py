from pathlib import Path

from CPDShell.Core.algorithms.GpraphCPD.Builders.matrix_builder import AdjacencyMatrixBuilder
from CPDShell.Core.algorithms.GpraphCPD.graph_cpd import GraphCPD
from CPDShell.generator.generator import ScipyDatasetGenerator
from CPDShell.generator.saver import DatasetSaver

# adj_matrix = np.array([
#     [0, 1, 1, 0],
#     [1, 0, 1, 1],
#     [1, 1, 0, 1],
#     [0, 1, 1, 0]
# ])
# graph = GraphMatrix(adj_matrix, 5)
# print(graph.sum_triangle())

path_string = "tests/test_CPDShell/test_configs/test_config_exp.yml"
distributions_name = "triangle"

saver = DatasetSaver(Path(), True)
generated = ScipyDatasetGenerator().generate_datasets(Path(path_string), saver)
data, expected_change_points = generated[distributions_name]
arg = 0.5
builder = AdjacencyMatrixBuilder(data, lambda a, b: abs(a - b) <= arg)
graph = builder.build_graph()
cpd = GraphCPD(graph)
# print(graph.sum_triangle())
# print(cpd.calculation_probability())
print(cpd.probability_diff())
# print(cpd.scrubber(10))
