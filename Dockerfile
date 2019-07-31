FROM julia:1.1
RUN apt-get install -y git hdf5-tools
RUN git clone https://github.com/climate-machine/Oceananigans.jl.git /Oceananigans.jl/
WORKDIR /Oceananigans.jl/
RUN julia --project -e "using Pkg; Pkg.instantiate();"
RUN julia --project -e "using Oceananigans"

