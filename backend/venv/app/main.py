from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from app import database, models, schemas, auth
from fastapi.middleware.cors import CORSMiddleware
# Create tables
models.Base.metadata.create_all(bind=database.engine)

app = FastAPI(title="LitScout Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # ✅ frontend URL
    allow_credentials=True,
    allow_methods=["*"],   # ✅ allow all HTTP methods
    allow_headers=["*"],   # ✅ allow all headers
)
# Dependency: DB session
def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Signup endpoint
@app.post("/signup", response_model=schemas.UserResponse)
def signup(user: schemas.UserCreate, db: Session = Depends(get_db)):
    # check if user already exists
    db_user = db.query(models.User).filter(models.User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    # ✅ hash password before saving
    hashed_pw = auth.hash_password(user.password)

    # ✅ save hashed password in DB
    new_user = models.User(
        full_name=user.full_name,
        email=user.email,
        password=hashed_pw      # store hashed_pw, not plain password
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

# Login endpoint
@app.post("/login")
def login(user: schemas.UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.email == user.email).first()
    if not db_user or not auth.verify_password(user.password, db_user.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = auth.create_access_token({"sub": db_user.email})
    return {"access_token": token, "token_type": "bearer"}

# About endpoint
@app.get("/about", response_model=schemas.AboutResponse)
def get_about():
    return {
        "mission": "LitScout automates systematic literature reviews using AI.",
        "how_it_works": [
            "Phase 1: Search & Filter with Semantic Scholar",
            "Phase 2: AI-Powered Screening with LLMs",
            "Phase 3: Analysis & Reporting (coming soon)"
        ]
    }