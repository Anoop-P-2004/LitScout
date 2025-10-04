# app/schemas.py
from pydantic import BaseModel, EmailStr
from typing import List


# =====================
# User-related Schemas
# =====================

# For signup request
class UserCreate(BaseModel):
    full_name: str
    email: EmailStr
    password: str


# For login request
class UserLogin(BaseModel):
    email: EmailStr
    password: str


# For response after signup
class UserResponse(BaseModel):
    id: int
    full_name: str
    email: EmailStr

    class Config:
        orm_mode = True


# =====================
# Token-related Schemas
# =====================

class Token(BaseModel):
    access_token: str
    token_type: str


# =====================
# About Endpoint Schema
# =====================

class AboutResponse(BaseModel):
    mission: str
    how_it_works: List[str]