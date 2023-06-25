FROM python:3.9

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY toxicity_checker backend/toxicity_checker
COPY bot.py backend/bot.py
COPY entrypoint.sh entrypoint.sh

WORKDIR backend
