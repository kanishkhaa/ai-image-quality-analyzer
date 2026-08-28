from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Text,
    DateTime
)

from datetime import datetime

from .database import Base


class Analysis(Base):

    __tablename__ = "analyses"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    filename = Column(
        String(255),
        nullable=False
    )

    quality_score = Column(
        Float,
        nullable=False
    )

    quality_label = Column(
        String(50),
        nullable=False
    )

    issues = Column(
        Text,
        nullable=False
    )

    statistics = Column(
        Text,
        nullable=False
    )

    explanation = Column(
        Text,
        nullable=False
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )