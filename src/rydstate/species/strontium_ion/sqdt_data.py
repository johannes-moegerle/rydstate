from pathlib import Path

from rydstate.qdt.sqdt import ModelSetSQDT, register_model_set_sqdt

DIRECTORY = Path(__file__).parent

# see sqdt_models/README.md for the references and how the quantum defects are derived from them
model_set_sqdt_strontium88_ion = ModelSetSQDT("Sr88_ion", models=DIRECTORY / "sqdt_models")
register_model_set_sqdt(model_set_sqdt_strontium88_ion)
