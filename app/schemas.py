from pydantic import BaseModel, Field, computed_field, field_validator, model_validator
from typing import Annotated
import re


class Message(BaseModel):
    text: Annotated[str, Field(..., description='enter your message and email')]
    has_currency: int = 0

    @model_validator(mode='before')
    @classmethod
    def detect_currency(cls, data):
        currency_symbols = ['£', '$', '€', '¥', '₹']
        text = data.get('text', '')

        data['has_currency'] = int(
            any(symbol in text for symbol in currency_symbols)
        )

        return data

    @field_validator('text')
    @classmethod
    def remove_currency(cls, value):
        return re.sub(r'[£$€¥₹]', '', value)

    @computed_field
    @property
    def char_count(self) -> int:
        return len(self.text)

    @computed_field
    @property
    def digit_count(self) -> int:
        return sum(1 for i in self.text if i.isdigit())

    @computed_field
    @property
    def uppercase_words(self) -> int:
        return sum(
            1 for i in self.text.split()
            if i.isupper() and len(i) > 1)

    @computed_field
    @property
    def has_url(self) -> int:
        return int(
            bool(
                re.search(r'http\S+|www\S+|\.com|\.co\.uk',self.text,re.IGNORECASE)))

    @computed_field
    @property
    def has_phone_num(self) -> int:
        return int(bool(re.search(r'\b\d{5,12}\b', self.text)))

