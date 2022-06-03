$ cd /Users/chrishill/projects/onan-brozonoyer-2022-06-03
$ git clone https://github.com/brozonoyer/Oceananigans.jl.git
$ cd Oceananigans.jl
$ git checkout cnh/addjld2
$ export JULIA_DEPOT_PATH=`pwd`/.julia
$ /Applications/Julia-1.7.app/Contents/Resources/julia/bin/julia --project=. 
julia> using Pkg
julia> Pkg.resolve()
julia> Pkg.instantiate()
julia> include("validation/barotropic_gyre/barotropic_gyre.jl")
