from sqlmodel import SQLModel, Field, Relationship, JSON
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone


class Student(SQLModel, table=True):
    __tablename__ = "student"
    id: Optional[int] = Field(default=None, primary_key=True)
    username: str = Field(index=True, unique=True)
    full_name: str | None = None
    hashed_password: str
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    
    # Relationship to MasteryRecord
    progress: List["MasteryRecord"] = Relationship(back_populates="student")

class StudyPlan(SQLModel, table=True):
    __tablename__ = "study_plans"
    id: Optional[int] = Field(default=None, primary_key=True)
    student_id: int = Field(foreign_key="student.id")
    plan: Dict[str, Any] = Field(default={}, sa_type=JSON) 
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

class MasteryRecord(SQLModel, table=True):
    __tablename__ = "mastery_records"
    id: Optional[int] = Field(default=None, primary_key=True)
    student_id: int = Field(foreign_key="student.id")
    topic: str
    score: float
    answers: Dict[str, Any] = Field(default={}, sa_type=JSON)
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    
    student: Student = Relationship(back_populates="progress")

class ChatHistory(SQLModel, table=True):
    __tablename__ = "chat_history"
    id: Optional[int] = Field(default=None, primary_key=True)
    student_id: int = Field(foreign_key="student.id")
    question: str
    answer: str
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )