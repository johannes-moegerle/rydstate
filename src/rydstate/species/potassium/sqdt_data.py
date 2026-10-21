from pathlib import Path

from rydstate.qdt.sqdt import ModelSetSQDT, register_model_set_sqdt

DIRECTORY = Path(__file__).parent

model_set_sqdt_potassium = ModelSetSQDT(
    "K",
    models=DIRECTORY / "sqdt_models",
    nist_data_file=DIRECTORY / "nist_data.txt",
)
register_model_set_sqdt(model_set_sqdt_potassium)
