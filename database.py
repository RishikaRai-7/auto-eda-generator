from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from datetime import datetime

DATABASE_URL = "sqlite:///./eda_reports.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)

class Base(DeclarativeBase):
    pass

class EDAReport(Base):
    __tablename__ = "reports"

    id         = Column(Integer, primary_key=True, index=True)
    filename   = Column(String)
    eda_json   = Column(Text)
    insights   = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)