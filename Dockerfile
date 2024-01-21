# The builder image, used to build the virtual environment
FROM python:3.11-buster as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    libatk-bridge2.0-0 \
    libxkbcommon0 \
    libatspi2.0-0 \
    libgbm1

# Install Poetry
RUN pip install poetry==1.4.2

# Set environment variables for Poetry
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Prepare the working directory
WORKDIR /app
COPY pyproject.toml poetry.lock ./

# Install Python dependencies including Playwright
RUN poetry install --without dev --no-root && rm -rf $POETRY_CACHE_DIR

# Install system-level dependencies required by Playwright
RUN playwright install-deps

# Install Playwright browsers
RUN poetry run playwright install

# The runtime image
FROM python:3.11-slim-buster as runtime

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

# Copy the virtual environment from the builder stage
COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}

# Copy the application code
COPY ./streamlit_agent ./streamlit_agent

# Run Streamlit
CMD ["streamlit", "run", "streamlit_agent/chat_pandas_df.py", "--server.port", "8051"]
