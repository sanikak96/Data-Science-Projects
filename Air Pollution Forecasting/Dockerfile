FROM jupyter/datascience-notebook

LABEL maintainer.name="Sanika Kulkarni" \
    maintainer.email="sanika.kulkarni@honeywell.com" \
    description="Jupyter notebook docker image with dependencies for Air Pollution Forecasting project"

# Install python dependencies
COPY requirements.txt /tmp/
RUN pip install --requirement /tmp/requirements.txt
