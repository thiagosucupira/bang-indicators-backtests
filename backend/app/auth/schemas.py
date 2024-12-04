from pydantic import BaseModel, EmailStr

class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str
    tier: str = "Noob"

class UserResponse(BaseModel):
    id: int
    username: str
    email: EmailStr
    tier: str

    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str