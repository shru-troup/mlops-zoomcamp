FROM agrigorev/zoomcamp-model:mlops-3.9.7-slim

WORKDIR /app
RUN pip install pipenv
COPY [ "Pipfile", "Pipfile.lock", "starter.py", "./" ]

RUN pipenv install --system --deploy
RUN mkdir /app/data

ENTRYPOINT [ "python", "starter.py", "--year",  "2021", "--month", "4" ]