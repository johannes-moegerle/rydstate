from rydstate.qdt.sqdt import ModelSetSQDT


class SQDTLithium(ModelSetSQDT):
    species = "Li"
    nist_data_file = "nist_data.txt"

    models_directory = "sqdt_models"
