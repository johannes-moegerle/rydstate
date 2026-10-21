from rydstate.qdt.sqdt import ModelSetSQDT


class SQDTCesium(ModelSetSQDT):
    species = "Cs"
    nist_data_file = "nist_data.txt"

    models_directory = "sqdt_models"
