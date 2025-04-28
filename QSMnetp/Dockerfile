FROM tensorflow/tensorflow:1.14.0-gpu

# Fix NVIDIA repository GPG key issue
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub && \
    apt-get update

# Install Python 3.7
RUN apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
    python3.7 \
    python3.7-dev \
    python3.7-distutils \
    && ln -s /usr/bin/python3.7 /usr/local/bin/python3.7 \
    && ln -sf /usr/bin/python3.7 /usr/local/bin/python3 \
    && ln -sf /usr/bin/python3.7 /usr/local/bin/python \
    && curl https://bootstrap.pypa.io/pip/3.7/get-pip.py -o get-pip.py \
    && python3.7 get-pip.py \
    && rm get-pip.py \
    && ln -sf /usr/local/bin/pip3.7 /usr/local/bin/pip



# Copy external code first (keeping original structure)
COPY external_code/Code /Code/
COPY external_code/Data /Data/
COPY external_code/Checkpoints /Checkpoints/

# Fix syntax error in network_model.py (change "reuse-reuse" to "reuse=reuse")
RUN sed -i 's/reuse-reuse/reuse=reuse/g' ./Code/network_model.py
RUN sed -i 's/from utils import \*/from utils import conv3d, deconv3d, max_pool, conv, batch_norm/g' ./Code/network_model.py
# Fix circular dependency
RUN sed -i '/from network_model import \*/d' ./Code/utils.py

# Copy application files to root
COPY api.py /Code/
COPY custom_inference.py /Code/
COPY requirements.txt /Code/

# Set up application structure
WORKDIR /Code/


# Set Python path to include current directory
ENV PYTHONPATH=/Code/

# Install requirements
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["python", "api.py"]

#docker build -t qsmnetp .
#docker run --rm --gpus all -p 5000:5000 -it qsmnetp
#python .\client_example.py --mat-file .\QSMnetp\external_code\Data\Test\Input\test_input1.mat --method mat