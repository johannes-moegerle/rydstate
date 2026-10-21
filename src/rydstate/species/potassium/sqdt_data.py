from rydstate.qdt.sqdt import ModelSetSQDT


class SQDTPotassium(ModelSetSQDT):
    species = "K"
    nist_data_file = "nist_data.txt"

    models_directory = "sqdt_models"
