from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from typing import Optional
import tempfile
from pathlib import Path

from fuse import predict

app = FastAPI()


def _load_from_s3(url: str) -> str:
    """Download an S3 object to a temporary file and return its path."""
    try:
        import boto3
        from urllib.parse import urlparse
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Missing dependency: {e}")

    parsed = urlparse(url)
    if parsed.scheme != "s3":
        raise HTTPException(status_code=400, detail="s3_url must start with s3://")
    bucket = parsed.netloc
    key = parsed.path.lstrip('/')
    s3 = boto3.client("s3")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(key).suffix or ".mp4")
    s3.download_fileobj(bucket, key, tmp)
    tmp.close()
    return tmp.name


@app.post("/score")
async def score(video: Optional[UploadFile] = File(None), s3_url: Optional[str] = Form(None)):
    """Return the deepfake risk score for an uploaded video or S3 URL."""
    if not video and not s3_url:
        raise HTTPException(status_code=400, detail="video or s3_url required")

    if video:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(video.filename).suffix)
        tmp.write(await video.read())
        tmp.close()
        path = tmp.name
    else:
        path = _load_from_s3(s3_url)

    score_value = predict(path)
    return {"risk_score": score_value}


def main() -> None:
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
