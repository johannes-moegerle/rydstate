from pathlib import Path

from rydstate.qdt.sqdt import ModelSetSQDT, register_model_set_sqdt

DIRECTORY = Path(__file__).parent

# Sr II (Sr+) energy levels, referenced to the Sr+ ground state (4p6.5s 2S1/2)
NIST_DATA_FILE = DIRECTORY / "nist_data.txt"

# see sqdt_models/README.md for the references and how the quantum defects are derived from them
model_set_sqdt_strontium88_ion = ModelSetSQDT(
    "Sr88_ion", models=DIRECTORY / "sqdt_models", nist_data_file=NIST_DATA_FILE
)
register_model_set_sqdt(model_set_sqdt_strontium88_ion)
