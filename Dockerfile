FROM tensorflow/tfx:1.4.0
WORKDIR ./
COPY ./pipeline ./pipeline
ENV PYTHONPATH="/pipeline:${PYTHONPATH}"
