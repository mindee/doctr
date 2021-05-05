# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from pydantic import BaseModel, Field


# Recognition output
class RecognitionOut(BaseModel):
    value: str = Field(..., example="Hello")
