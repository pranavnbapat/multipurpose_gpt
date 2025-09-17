# app/models/enums.py

from enum import Enum

class ModelName(str, Enum):
    gpt_4o_mini = "gpt-4o-mini"
    gpt_4o = "gpt-4o"
    gpt_5_mini = "gpt-5-mini"
    gpt_5 = "gpt-5"

