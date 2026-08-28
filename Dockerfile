# Single image, two Cloud Run services. main.py (public API) and
# worker.py (Cloud Tasks push target, no public invoker) are deployed
# from this exact same image — only the container's start command
# differs between the two `gcloud run deploy` calls in README.md's
# deploy section. Keeps build/deploy simple (one image to build, test,
# and version) rather than maintaining two near-identical Dockerfiles.

FROM python:3.12-slim

WORKDIR /app

# Keeps the image small and the build cache useful — requirements.txt
# rarely changes as often as the actual code below it.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Cloud Run sets $PORT itself (defaults to 8080) — respected here rather
# than hardcoded, so this image runs correctly both on Cloud Run and
# locally (`docker run -e PORT=8080 ...`).
ENV PORT=8080
EXPOSE 8080

# Default target is the public API service (main.py). Override at deploy
# time for the worker service:
#   gcloud run deploy queryagent-worker --image <this image> \
#     --command python3 --args="-m,uvicorn,worker:app,--host,0.0.0.0,--port,8080"
# (see README.md's "Deploying to Cloud Run + Cloud Tasks" section for
# the full two-service deploy, including the worker's no-public-invoker
# IAM binding).
CMD ["sh", "-c", "python3 -m uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
