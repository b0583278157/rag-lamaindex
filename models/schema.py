from pydantic import BaseModel
from typing import Optional, List
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime


class SourceFile(BaseModel):
    path: str
    last_modified: Optional[str] = None


class SourceInfo(BaseModel):
    tool: str
    root_path: str
    files: List[SourceFile]


class Source(BaseModel):
    file: str
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    anchor: Optional[str] = None


class Rule(BaseModel):
    id: str
    rule: str
    scope: Optional[str] = None
    notes: Optional[str] = None
    category: Optional[str] = None
    source: Optional[Source] = None
    observed_at: Optional[str] = None


class Decision(BaseModel):
    id: str
    title: str
    summary: str
    source: Optional[Source] = None
    observed_at: Optional[str] = None


class WarningItem(BaseModel):
    id: str
    message: str
    severity: str
    source: Optional[Source] = None
    observed_at: Optional[str] = None


class KnowledgeData(BaseModel):
    schema_version: str = "1.0"
    decisions: List[Decision] = []
    rules: List[Rule] = []
    warnings: List[WarningItem] = []