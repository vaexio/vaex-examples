FROM condaforge/mambaforge:4.9.2-5 as conda

# Copy relevant files & config & training package
COPY env.yml app.py ./

# Create the environment and install the training package
RUN mamba env create -f env.yml
RUN echo "source activate training-image" > ~/.bashrc \
    && conda clean --all --yes
ENV PATH /opt/conda/envs/training-image/bin:$PATH

# Expose the relevant port
EXPOSE 8000

# Define an entry point: a bash shell
ENTRYPOINT ["/bin/bash", "-c"]

# Run app.py through gunicorn, with 5 async workers
CMD ["python app.py"]
