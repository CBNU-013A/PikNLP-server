# Base Image
FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime

# Poetry 설치
RUN pip install -U poetry

# 경로 정의
WORKDIR /workdir

# 로컬에 있는 pyproject.toml, poetry.lock 파일을 컨테이너로 복사
COPY poetry.lock pyproject.toml /workdir/

# Poetry를 이용하여 의존성 설치 
RUN poetry config virtualenvs.create false \
 && poetry install --no-root --no-interaction 

# 로컬에 있는 소스코드를 컨테이너로 복사
COPY . /workdir

# Python 경로 설정
# ENV PYTHONPATH=/usr/local/bin/python3.12

# Poetry 바이너리 권한 확인 및 설정
# RUN chmod +x /usr/local/bin/poetry

# # Poetry가 설치된 Python을 사용하도록 설정
# RUN sed -i '1s|^.*$|#!/usr/local/bin/python3.12|' /usr/local/bin/poetry

# # 권한과 바이너리 위치 확인
# RUN ls -l /usr/local/bin/poetry

# Port Expose
EXPOSE 18000

# command to run
CMD ["poetry", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "18000"]