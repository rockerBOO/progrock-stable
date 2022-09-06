FROM nvidia/cuda:11.3.1-base-ubuntu20.04

SHELL ["/bin/bash", "-c"]

RUN apt-get update \
 && apt-get install --no-install-recommends -y curl wget git \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/* 

RUN wget --progress=dot:giga https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh -O ~/miniconda.sh \
 && bash ~/miniconda.sh -b -p $HOME/miniconda \
 && $HOME/miniconda/bin/conda init

COPY . /root/prs

WORKDIR /root/prs

RUN eval "$($HOME/miniconda/bin/conda shell.bash hook)" \
 && conda env create -f environment.yaml \
 && conda activate prs \
 && pip install --no-cache-dir gradio==3.1.7

VOLUME /root/.cache
VOLUME /data
VOLUME /output

ENV PYTHONUNBUFFERED=1
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860
EXPOSE 7860

RUN ln -s /data /root/prs/models \
 && mkdir -p /output /root/prs/out \
 && ln -s /output /root/prs/out

WORKDIR /root/prs

ENTRYPOINT ["/root/prs/docker-bootstrap.sh"]
CMD python prs.py
