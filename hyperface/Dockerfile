#
# BUILD
# sudo docker build -t hyperface .
#
# RUN
# sudo nvidia-docker run -it --rm -p 8888-9000:8888-9000 -v /home/lmiguel/Projects:/Projects --name hyperface hyperface /bin/bash
#
# REMOVE
# sudo docker rm hyperface
#
# Based on https://github.com/pytorch/pytorch/blob/master/Dockerfile
# Based on https://github.com/cmusatyalab/openface/blob/master/opencv-dlib-torch.Dockerfile
#

FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04

RUN echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

#
# Install essential libraries and Python 2
#
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    curl \
    gfortran \
    git \
    graphicsmagick \
    libgraphicsmagick1-dev \
    libatlas-dev \
    libavcodec-dev \
    libavformat-dev \
    libboost-all-dev \
    libgtk2.0-dev \
    libjpeg-dev \
    liblapack-dev \
    libswscale-dev \
    pkg-config \
    python-dev \
    python-numpy \
    python-protobuf\
    software-properties-common \
    zip \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*


#
# Install Torch
#
# # deprecated instructions
RUN apt-get update && \
    apt-get -y install sudo
# RUN curl -s https://raw.githubusercontent.com/torch/ezinstall/master/install-deps | bash -e
RUN git clone https://github.com/torch/distro.git ~/torch --recursive
RUN cd ~/torch && bash install-deps && ./install.sh && \
    cd install/bin && \
    ./luarocks install nn && \
    ./luarocks install dpnn && \
    ./luarocks install image && \
    ./luarocks install optim && \
    ./luarocks install csvigo && \
    ./luarocks install torchx && \
    ./luarocks install tds

#
# Install OpenCV 2.4.11
#
RUN cd ~ && \
    mkdir -p ocv-tmp && \
    cd ocv-tmp && \
    curl -L https://github.com/opencv/opencv/archive/2.4.13.4.zip -o ocv.zip && \
    unzip ocv.zip && \
    cd opencv-2.4.13.4 && \
    mkdir release

RUN cd ~/ocv-tmp/opencv-2.4.13.4/release && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D BUILD_PYTHON_SUPPORT=ON -D CUDA_GENERATION=Kepler .. && \
    make -j8 && \
    make install && \
    rm -rf ~/ocv-tmp


#
# Install dlib 19.0
#
RUN cd ~ && \
    mkdir -p dlib-tmp && \
    cd dlib-tmp && \
    curl -L \
         https://github.com/davisking/dlib/archive/v19.0.tar.gz \
         -o dlib.tar.bz2 && \
    tar xf dlib.tar.bz2 && \
    cd dlib-19.0/python_examples && \
    mkdir build && \
    cd build && \
    cmake ../../tools/python && \
    cmake --build . --config Release && \
    cp dlib.so /usr/local/lib/python2.7/dist-packages && \
    rm -rf ~/dlib-tmp

#
# hyperface requirements
#
# install pip
RUN curl -o get-pip.py https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py
#
# install chainer, flask, matplotlib
RUN pip install chainer==1.17.0 && \
    pip install flask && \
    pip install flask-socketio && \
    pip install matplotlib

#
# Install Jupyter
#
RUN pip install jupyter && \
    echo "alias start-jupyter=\"jupyter notebook --notebook-dir=. --ip=0.0.0.0 --no-browser --allow-root\"" >> ~/.bashrc

#
# working directory
#
WORKDIR /Projects

#
# # RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-4.2.12-Linux-x86_64.sh  && \
# #      chmod +x ~/miniconda.sh && \
# #      ~/miniconda.sh -b -p /opt/conda && \
# #      rm ~/miniconda.sh && \
# #      /opt/conda/bin/conda install conda-build && \
# #      /opt/conda/bin/conda create -y --name pytorch-py35 python=3.5.2 numpy pyyaml scipy ipython mkl&& \
# #      /opt/conda/bin/conda clean -ya
# # ENV PATH /opt/conda/envs/pytorch-py35/bin:$PATH
# # RUN conda install --name pytorch-py35 -c soumith magma-cuda80
# # RUN conda install pytorch torchvision cuda80 -c soumith
# #
# # # Install face-alignment package
# # WORKDIR /workspace
# # RUN chmod -R a+w /workspace
# # RUN git clone https://github.com/1adrianb/face-alignment
# # WORKDIR /workspace/face-alignment
# # RUN pip install -r requirements.txt
# # COPY ./setup.py /workspace/face-alignment/setup.py
# # RUN python setup.py install
# #
# # # install Jupyter
# # RUN pip install jupyter
# # # add alias to start jupyter
# # COPY ./aliases.append /tmp
# # RUN cat /tmp/aliases.append >> ~/.bashrc && \
# # 	rm -f /tmp/aliases.append
# #
# # # install pytorch from http://pytorch.org/
# # RUN pip install http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp35-cp35m-manylinux1_x86_64.whl &&  pip install notebook ipywidgets
# # # install libSM.so.6
# # RUN apt-get update && apt-get install -y python-qt4
# #
# # # copy models and configurations
# # COPY ./jupyter_notebook_config.py /root
# # RUN mkdir /root/.face_alignment
# # COPY ./data /root/.face_alignment/data
# #
# # # working directory
# # WORKDIR /Projects
