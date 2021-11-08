# # Tilted bottom boundary layer example
#
# This example simulates a two-dimensional tilted oceanic bottom boundary layer based 
# on Wenegrat et al. (2020). It demonstrates how to tilt the domain by
#
#   * Changing the direction of the buoyancy acceleration
#   * Changing the axis of rotation for Coriolis
#
# ## Install dependencies
#
# First let's make sure we have all required packages installed.

# ```julia
# using Pkg
# pkg"add Oceananigans, Plots"
# ```

# ## Load `Oceananigans.jl` and define problem constants
#
# Since the simulation is taken from Wenegrat et al. (2020), we can set a named tuple
# with every simulation parameter that we use:

using Oceananigans
using Oceananigans.Units
using Printf
using CUDA

params = (f₀ = 1e-4, #1/s
          V∞ = 0.1, # m/s
          N²∞ = 1e-5, # 1/s²
          θ_rad = 0.05,
          Lx = 1000, # m
          Lz = 100, # m
          Nx = 64,
          Nz = 64,
          ν = 5e-4, # m²/s
          sponge_frac = 1/5,
          sponge_rate = √1e-5, # 1/s
          z_0 = 0.1, # m (roughness length)
          )

arch = CPU()
ĝ = [sin(params.θ_rad), 0, cos(params.θ_rad)]


# Here `f₀` is the Coriolis frequency, `V∞` in the interior `v`-velocity, `N²∞` is the
# interior stratification, `θ_rad` is the bottom slope in radians, `ν` is the eddy viscosity,
# and `z_0` is rthe roughness length (needed for the drag at the bottom).
#
# ## Instantiating and configuring a model
#
# A core Oceananigans type is `NonhydrostaticModel`. We build an `NonhydrostaticModel`
# by passing it a `grid`, plus information about the equations we would like to solve.
#

