from pathlib import Path

from rydstate.qdt.sqdt import ModelSetSQDT, register_model_set_sqdt

DIRECTORY = Path(__file__).parent

# the directory only contains a README, i.e. hydrogen has zero quantum defects
model_set_sqdt_hydrogen = ModelSetSQDT("H", models=DIRECTORY / "sqdt_models")
register_model_set_sqdt(model_set_sqdt_hydrogen)

model_set_sqdt_hydrogen_textbook = ModelSetSQDT("H_textbook", models=DIRECTORY / "sqdt_models")
register_model_set_sqdt(model_set_sqdt_hydrogen_textbook)
