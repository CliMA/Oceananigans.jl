ENV["GKSwstype"] = "nul"
using Plots

using Printf
using Statistics

using Oceananigans
using Oceananigans.Advection
using Oceananigans.AbstractOperations
using Oceananigans.OutputWriters
using Oceananigans.Grids
using Oceananigans.Fields
using Oceananigans.Utils: prettytime

using Oceananigans.Diagnostics: accurate_cell_advection_timescale

using Oceananigans.Models.HydrostaticFreeSurfaceModels: VectorInvariant
using Oceananigans.OutputReaders: FieldTimeSeries, @compute

using Oceananigans.Advection: ZWENO, WENOVectorInvariant
#####
##### The Bickley jet
#####

Ψ(y) = - tanh(y)
U(y) = sech(y)^2

# A sinusoidal tracer
C(y, L) = sin(2π * y / L)

# Slightly off-center vortical perturbations
ψ̃(x, y, ℓ, k) = exp(-(y + ℓ/10)^2 / 2ℓ^2) * cos(k * x) * cos(k * y)

# Vortical velocity fields (ũ, ṽ) = (-∂_y, +∂_x) ψ̃
ũ(x, y, ℓ, k) = + ψ̃(x, y, ℓ, k) * (k * tan(k * y) + y / ℓ^2) 
ṽ(x, y, ℓ, k) = - ψ̃(x, y, ℓ, k) * k * tan(k * x) 

"""
    run_bickley_jet(output_time_interval = 2, stop_time = 200, arch = CPU(), Nh = 64, ν = 0,
                    momentum_advection = VectorInvariant())

Run the Bickley jet validation experiment until `stop_time` using `momentum_advection`
scheme or formulation, with horizontal resolution `Nh`, viscosity `ν`, on `arch`itecture.
"""
function run_bickley_jet(C3₀, C3₁; arch = GPU(), Nh = 64)

    C3₂ = 1 - C3₀ - C3₁

    grid = RectilinearGrid(arch, size=(Nh, Nh, 1),
                                x = (-2π, 2π), y=(-2π, 2π), z=(0, 1), halo = (4, 4, 4),
                                topology = (Periodic, Periodic, Bounded))

    model = HydrostaticFreeSurfaceModel(momentum_advection = WENO5((C3₀, C3₁, C3₂), vector_invariant=true),
                                                      grid = grid,
                                                   tracers = (),
                                                   closure = nothing,
                                              free_surface = ImplicitFreeSurface(gravitational_acceleration=10.0),
                                                  coriolis = nothing,
                                                  buoyancy = nothing)

    # ** Initial conditions **
    #
    # u, v: Large-scale jet + vortical perturbations
    #    c: Sinusoid

    # Parameters
    ϵ = 0.1 # perturbation magnitude
    ℓ = 0.5 # Gaussian width
    k = 0.5 # Sinusoidal wavenumber

    # Total initial conditions
    uᵢ(x, y, z) = U(y) + ϵ * ũ(x, y, ℓ, k)
    vᵢ(x, y, z) = ϵ * ṽ(x, y, ℓ, k)
    # cᵢ(x, y, z) = C(y, grid.Ly)

    set!(model, u=uᵢ, v=vᵢ) #, c=cᵢ)

    Δt = 0.1 * accurate_cell_advection_timescale(grid, model.velocities)

    nsteps  = floor(Int, 200 / Δt)
    ninterv = floor(Int, 10 / Δt)
    
    nsave = floor(Int, nsteps/ninterv)

    u, v, w = model.velocities

    enst   = zeros(nsave)

    @info "Running a simulation of an unstable Bickley jet with $(Nh)² degrees of freedom..."

    time = 0
    for i in 1:nsave
        enst[i] = sum((∂x(v) - ∂y(u))^ 2) / Nh^2
        for j in 1:ninterv 
            time_step!(model, Δt)
            time = time + Δt
        end
        @info "saving iteration number $i, with time $time, and var $(enst[i])"
    end

    @show model.clock.time

    return (model, enst)
end
   
N = 10
# timestep size (h = 1/N implies T=1 at the final time)
h = 1 / N

Nparticles = 20
# number of ensemble members
J = Nparticles * 2


priors = vcat([rand() - 0.2 for i in 1:Nparticles], [rand() + 0.1 for i in 1:Nparticles])


# function forward_map(enst,)

# end


# function