from rydstate.qdt.sqdt import ModelSetSQDT


class SQDTHydrogen(ModelSetSQDT):
    species = "H"

    models_directory = "sqdt_models"


class SQDTHydrogenTextBook(ModelSetSQDT):
    species = "H_textbook"

    models_directory = "sqdt_models"
