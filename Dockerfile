FROM python:3.12-slim

WORKDIR /claude-code-proxy

ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1

# Install dependencies from the pinned requirements file.
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
	&& pip install --no-cache-dir -r requirements.txt

# Copy the application code.
COPY . .

# Start the proxy.
EXPOSE 8082
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8082", "--log-level", "warning"]
