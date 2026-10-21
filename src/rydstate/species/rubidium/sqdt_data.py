from rydstate.qdt.sqdt import ModelSetSQDT


class SQDTRubidium(ModelSetSQDT):
    species = "Rb"
    nist_data_file = "nist_data.txt"

    models_directory = "sqdt_models"
