from fastapi import FastAPI

app = FastAPI(title="hit-api")


@app.get("/")
def root():
    # Root Endpoint 
    return {"message": "Hello!! I am new to Docker and Tech in!"}


@app.get("/health")
def health():
    return {"status": "ok"}