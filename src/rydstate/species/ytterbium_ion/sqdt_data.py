from pathlib import Path

from rydstate.qdt.sqdt import ModelSetSQDT, register_model_set_sqdt

DIRECTORY = Path(__file__).parent

# only the NIST levels, i.e. no quantum defects, see sqdt_models/README.md
model_set_sqdt_ytterbium174_ion = ModelSetSQDT("Yb174_ion", models=DIRECTORY / "sqdt_models")
register_model_set_sqdt(model_set_sqdt_ytterbium174_ion)
