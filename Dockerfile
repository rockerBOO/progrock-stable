FROM nvidia/cuda:11.3.1-base-ubuntu20.04

SHELL ["/bin/bash", "-c"]

# ---

COPY . /root/prs

# RUN groupadd -r prs && useradd --no-log-init -r -g prs prs

# USER prs

VOLUME /root/.cache
VOLUME /models
VOLUME /output
VOLUME /settings/user_settings.json
VOLUME /realesrgan-ncnn-vulkan
VOLUME /psr.py
VOLUME /input/init_image.img
VOLUME /root/prs

WORKDIR /root/prs

RUN ls -lah
# RUN ls

# RUN chmod +x realesrgan-ncnn-vulkan.yo

RUN ln -s /models models \
 && mkdir -p /output out \
 && ln -s /output out

# RUN rm prs.py && ln -s /prs.py prs.py

# ---


RUN apt-get update \
 && apt-get install --no-install-recommends -y \
 curl=7.68.0-1ubuntu2.13 \
 wget=1.20.3-1ubuntu2 \
 git=1:2.25.1-1ubuntu3.5 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/* 

# --- 
# Vulkan requirements for realesrgan-ncnn-vulkan
# ---

# ENV TZ=America/New_York
# RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# SHELL ["/bin/bash", "-o", "pipefail", "-c"]
# RUN wget -qO - http://packages.lunarg.com/lunarg-signing-key-pub.asc | apt-key add - \
#   && wget -qO /etc/apt/sources.list.d/lunarg-vulkan-focal.list http://packages.lunarg.com/vulkan/lunarg-vulkan-focal.list 

# RUN apt-get update \
#  && apt-get install --no-install-recommends -y \
#  libgomp1=10.3.0-1ubuntu1~20.04 \
#  vulkan-sdk=1.3.224.0~rc2-1lunarg20.04-1 \
#  && apt-get clean \
#  && rm -rf /var/lib/apt/lists/* 

# --- 

RUN wget --progress=dot:giga https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh -O ~/miniconda.sh \
 && bash ~/miniconda.sh -b -p $HOME/miniconda \
 && $HOME/miniconda/bin/conda init

RUN eval "$($HOME/miniconda/bin/conda shell.bash hook)" \
 && conda env create -f environment.yaml \
 && conda activate prs \
 && pip install --no-cache-dir gradio==3.1.7

ENV PYTHONUNBUFFERED=1

# If using Gradio 
# ENV GRADIO_SERVER_NAME=0.0.0.0
# ENV GRADIO_SERVER_PORT=7860
# EXPOSE 7860

ENTRYPOINT ["/root/prs/docker-bootstrap.sh"]
CMD python prs.py -s /settings/user_settings.json
