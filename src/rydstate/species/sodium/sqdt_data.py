from rydstate.qdt.sqdt import ModelSetSQDT


class SQDTSodium(ModelSetSQDT):
    species = "Na"
    nist_data_file = "nist_data.txt"

    models_directory = "sqdt_models"
