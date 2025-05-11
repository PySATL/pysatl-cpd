from new_pysatl_cpd.logger import cpd_logger
from new_pysatl_cpd.steps.report_generation_step.report_builders.report_builder import ReportBuilder
from new_pysatl_cpd.steps.report_generation_step.report_visualizers.report_visualizer import ReportVisualizer


class DummyReportVisualizer(ReportVisualizer):
    def draw(self, report_builder: ReportBuilder) -> None:
        cpd_logger.debug("DummyReportVisualizer draw method")
        path = self.path_to_save
        with open(f"{path}/result.txt", "w", encoding="utf-8") as file:
            file.write("first string.\n")
            file.write("second string.\n")
