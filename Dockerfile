FROM dolfinx/dolfinx:v0.9.0

# Set workspace inside container
WORKDIR /workspace

# System dependencies (OpenGL, visualization, etc.)
RUN apt update && \
    apt upgrade -y && \
    apt install -y \
        texlive-latex-base dvipng texlive-latex-recommended texlive-fonts-recommended cm-super\
        texlive-latex-extra \
        xvfb \
        libgl1-mesa-dri \
        libglx-mesa0 \
        libglu1-mesa \
        mesa-utils && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Python packages (add spyder-kernels!)
RUN pip3 install --no-cache-dir \
    pandas \
    imageio \
    pyvista \
    scipy \
    mpi4py \
    petsc4py \
    numpy \
    matplotlib \
    spyder-kernels==3.1.* \
    h5py \
    jupyter
    
# Make port 8888 available to the world outside this container
EXPOSE 8888

# Define environment variable
ENV NAME World

