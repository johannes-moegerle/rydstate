from typing import ClassVar

from rydstate.qdt.sqdt import ModelSetSQDT


class SQDTHydrogen(ModelSetSQDT):
    species = "H"
    is_default = True

    models: ClassVar = []


class SQDTHydrogenTextBook(ModelSetSQDT):
    species = "H_textbook"
    is_default = True

    models: ClassVar = []
