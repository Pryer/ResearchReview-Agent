"""解析后文本的章节区间；不表示 PDF 空间坐标。"""
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


class SectionSpan(BaseModel):
    section: str
    page: Optional[int] = Field(default=None, ge=1)
    text: str
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    offset_basis: Literal["page", "full_text"] = "page"

    @model_validator(mode="after")
    def valid_range(self):
        if self.char_end <= self.char_start or self.char_end - self.char_start != len(self.text):
            raise ValueError("章节区间与原文长度不一致")
        if (self.page is None) != (self.offset_basis == "full_text"):
            raise ValueError("章节页码与位置基准不一致")
        return self
