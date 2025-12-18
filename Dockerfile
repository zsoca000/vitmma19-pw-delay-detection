FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

# Install system essentials
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install PyG Acceleration (Sensitive versions)
RUN pip install --no-cache-dir \
    torch-geometric \
    pyg_lib \
    torch_scatter \
    torch_sparse \
    torch_cluster \
    torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.6.0+cu124.html

# Install standard requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your code folders
COPY src/ ./src/
COPY run.sh .

# Prepare the execution script
RUN mkdir shared
RUN chmod +x run.sh

# Expose TensorBoard port
EXPOSE 6006

CMD ["./run.sh"]