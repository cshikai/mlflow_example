FROM python:3.12.4
RUN python3 -m pip install --user pipx \
&& python3 -m pipx ensurepath\
&& python3 -m pipx install poetry  \
&& /bin/bash -c 'source ~/.bashrc; poetry config virtualenvs.create false' 
COPY build/ /ml_project/build
WORKDIR /ml_project/build
RUN /bin/bash -c 'source ~/.bashrc; poetry install'
WORKDIR /ml_project/
COPY src/ /ml_project/src
