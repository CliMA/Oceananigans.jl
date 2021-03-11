using DataDeps
using JLD2

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

dd = DataDep("cubed_sphere_32_grid",
    "Conformal cubed sphere grid with 32×32 grid points on each face",
    "https://github.com/CliMA/OceananigansArtifacts.jl/raw/main/cubed_sphere_grids/cubed_sphere_32_grid.jld2",
    "3cc5d86290c3af028cddfa47e61e095ee470fe6f8d779c845de09da2f1abeb15" # sha256sum
)

DataDeps.register(dd)

cs32 = jldopen(datadep"cubed_sphere_32_grid/cubed_sphere_32_grid.jld2")
