FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY wyoming_nexa_bridge.py .

# Wyoming default port for STT
EXPOSE 10300

# Default: Spanish, tablet IP must be set via environment variable
ENV NEXA_URL=http://192.168.1.100:8080
ENV LANGUAGE=es

# Use shell form so env vars are expanded at runtime
CMD python wyoming_nexa_bridge.py --nexa-url "$NEXA_URL" --language "$LANGUAGE"
