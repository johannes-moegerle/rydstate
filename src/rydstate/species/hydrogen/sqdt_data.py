from typing import ClassVar

from rydstate.qdt.sqdt import ModelSetSQDT


class SQDTHydrogen(ModelSetSQDT):
    species = "H"

    models: ClassVar = []


class SQDTHydrogenTextBook(ModelSetSQDT):
    species = "H_textbook"

    models: ClassVar = []
