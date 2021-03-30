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

## Model setup

model = HydrostaticFreeSurfaceModel(
          architecture = CPU(),
                  grid = grid,
    momentum_advection = VectorInvariant(),
          free_surface = ExplicitFreeSurface(gravitational_acceleration=0.1),
        # free_surface = ImplicitFreeSurface(gravitational_acceleration=0.1)
              coriolis = nothing,
            # coriolis = HydrostaticSphericalCoriolis(scheme = VectorInvariantEnstrophyConserving()),
               closure = nothing,
               tracers = nothing,
              buoyancy = nothing
)

## Very small sea surface height perturbation so the resulting dynamics are well-described
## by a linear free surface.

A  = 1e-5 * H  # Amplitude of the perturbation
λ₀ = 0   # Central longitude
φ₀ = 40  # Central latitude
Δλ = 20  # Longitudinal width
Δφ = 20  # Latitudinal width

η′(λ, φ, z) = A * exp(- (λ - λ₀)^2 / Δλ^2) * exp(- (φ - φ₀)^2 / Δφ^2)

# set!(model, η=η′)

set!(model.free_surface.η, η′)
