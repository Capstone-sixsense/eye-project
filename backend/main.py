from fastapi import FastAPI


# Temporary placeholder app so the backend container can start before
# the real API and model integration are implemented.
app = FastAPI(title="eye-project backend")


@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "temporary backend is running"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
