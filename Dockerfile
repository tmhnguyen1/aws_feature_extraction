FROM python:3.7-slim-buster

# install dependencies
RUN pip3 install pandas numpy tsfresh

# setup env variables
ENV PYTHONUNBUFFERED=TRUE

ENTRYPOINT [ "python3" ]