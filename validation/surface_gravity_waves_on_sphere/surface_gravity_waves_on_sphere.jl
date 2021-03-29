using Statistics
using Logging
using Printf
using DataDeps
using JLD2

using Oceananigans
using Oceananigans.Units
using Oceananigans.CubedSpheres
using Oceananigans.Coriolis
using Oceananigans.Models.HydrostaticFreeSurfaceModels
using Oceananigans.TurbulenceClosures

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

Logging.global_logger(OceananigansLogger())

dd = DataDep("cubed_sphere_32_grid",
    "Conformal cubed sphere grid with 32×32 grid points on each face",
    "https://github.com/CliMA/OceananigansArtifacts.jl/raw/main/cubed_sphere_grids/cubed_sphere_32_grid.jld2",
    "3cc5d86290c3af028cddfa47e61e095ee470fe6f8d779c845de09da2f1abeb15" # sha256sum
)

DataDeps.register(dd)

cs32_filepath = datadep"cubed_sphere_32_grid/cubed_sphere_32_grid.jld2"

H = 4kilometers
grid = ConformalCubedSphereGrid(cs32_filepath, Nz=1, z=(-H, 0))

