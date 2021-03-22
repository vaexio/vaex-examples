FROM condaforge/mambaforge:4.9.2-5 as conda

# Copy relevant files & config & training package
COPY env.yml setup.py ./
ADD har_model ./har_model/

# Create the environment and install the training package
RUN mamba env create -f env.yml
RUN echo "source activate training-image" > ~/.bashrc \
    && conda clean --all --yes
ENV PATH /opt/conda/envs/training-image/bin:$PATH
RUN pip install -e .

# Run the container
ENTRYPOINT ["python", "har_model/train.py"]
# ENTRYPOINT ["/bin/bash", "-c"]
