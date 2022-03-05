FROM julia:1.7.2
LABEL maintainer="Ali Ramadhan <alir@mit.edu>"

RUN apt-get update && apt-get install -y hdf5-tools
COPY . /Oceananigans.jl/

WORKDIR /Oceananigans.jl/
RUN julia --project -e "using Pkg; Pkg.instantiate();"
RUN julia --project -e "using Oceananigans"
