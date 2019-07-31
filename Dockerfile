FROM julia:1.1
LABEL maintainer="alir@mit.edu"

RUN apt-get update && apt-get install -y git hdf5-tools
RUN git clone https://github.com/climate-machine/Oceananigans.jl.git /Oceananigans.jl/

WORKDIR /Oceananigans.jl/
RUN julia --project -e "using Pkg; Pkg.instantiate();"
RUN julia --project -e "using Oceananigans"

