FROM apache/airflow:2.9.0-python3.11

USER root

# The DAG executes Module 1 directly, so its pinned runtime must be present in
# the Airflow image rather than relying on a path from the host machine.
COPY requirements.txt /tmp/requirements.txt
COPY module_01_model_optimization/ /opt/smartfinance/module_01_model_optimization/
RUN python -m venv /opt/smartfinance/module_01_venv \
    && /opt/smartfinance/module_01_venv/bin/pip install --no-cache-dir \
        -r /tmp/requirements.txt \
    && chown -R airflow:root /opt/smartfinance

USER airflow
