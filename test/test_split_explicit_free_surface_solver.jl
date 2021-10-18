include(pwd() * "/src/Models/HydrostaticFreeSurfaceModels/split_explicit_free_surface.jl")
const g_Earth = 9.80665

arch = Oceananigans.GPU()
FT = Float64
topology = (Periodic, Periodic, Bounded)
Nx = Ny = Nz = 16
Lx = Ly = Lz = 2π
grid = RegularRectilinearGrid(topology=topology, size=(Nx, Ny, Nz), x=(0, Lx), y=(0, Ly), z=(-Lz, 0))

tmp = SplitExplicitFreeSurface()
sefs = SplitExplicitState(grid, arch)
sefs = SplitExplicitForcing(grid, arch)
sefs = SplitExplicitFreeSurface(grid, arch)

sefs.Gᵁ
sefs.η .= 0.0
sefs.state.η === sefs.η
sefs.forcing.Gᵁ === sefs.Gᵁ

#=
∂t(η) = -∇⋅U⃗ 
∂t(U⃗) = - ∇η + f⃗
divᶜᶜᶜ(U⃗)
=#

using Oceananigans.Utils
using Oceananigans.Operators
using KernelAbstractions

@kernel function free_surface_substep_kernel!(grid, Δτ, η, U, V, Gᵁ, Gⱽ, η̅, U̅, V̅, velocity_weight, tracer_weight)
    i, j = @index(Global, NTuple)
    # ∂τ(η) = - ∇⋅U⃗
    @inbounds η[i, j, 1] -=  Δτ * div_xyᶜᶜᵃ(i, j, 1, grid, U, V)
    @synchronize # since ∇η uses nonlocal threads
    # ∂τ(U⃗) = - ∇η + G⃗
    @inbounds U[i, j, 1] +=  Δτ * (-∂xᶠᶜᵃ(i, j, 1,  grid, η) + Gᵁ[i, j, 1])
    @inbounds V[i, j, 1] +=  Δτ * (-∂yᶜᶠᵃ(i, j, 1,  grid, η) + Gⱽ[i, j, 1])
    # time-averaging
    @inbounds U̅[i, j, 1] +=  velocity_weight * U[i, j, 1]
    @inbounds V̅[i, j, 1] +=  velocity_weight * V[i, j, 1]
    @inbounds η̅[i, j, 1] +=  tracer_weight * η[i, j, 1]
end

# TODO Have location for vertically integrated tendencies
U, V, η̅, U̅, V̅, Gᵁ, Gⱽ  = sefs.U, sefs.V, sefs.η̅, sefs.U̅, sefs.V̅, sefs.Gᵁ, sefs.Gⱽ
η = sefs.η
velocity_weight = 0.0
tracer_weight = 0.0
Δτ = 1.0

# set!(η, f(x,y))
η₀(x,y) = sin(x)
set!(η, η₀)

U₀(x,y) = cos(x)
set!(U, U₀)


V₀(x,y) = 0.0
set!(V, V₀)

function substep!(arch, grid, model, split_explicit_free_surface::SplitExplicitFreeSurface)
    # unpack
    sefs = split_explicit_free_surface
    η, U, V, η̅, U̅, V̅, = sefs.U, sefs.V, sefs.η̅, sefs.U̅, sefs.V̅, sefs.U, sefs.V
    Gᵁ, Gⱽ = sefs.Gᵁ, sefs.Gⱽ
    substeps = sefs.settings.substeps
    velocity_weights = sefs.settings.velocity_weights
    tracer_weights = sefs.settings.tracer_weights

    # TODO: DEFINE Δτ
    Δτ = 2*model.Δt / substeps # go twice as far for averaging

    for i in 1:substeps
        velocity_weight = velocity_weights[i]
        tracer_weight = tracer_weights[i]
        # fill_halo_regions! is blocking 
        fill_halo_regions!(arch, grid, η)
        fill_halo_regions!(arch, grid, U)
        fill_halo_regions!(arch, grid, V)
        # substep 
        event = launch!(arch, grid, :xy, free_surface_substep_kernel!, 
                        grid, Δτ, 
                        η.data, U.data, V.data, Gᵁ.data, Gⱽ.data, 
                        η̅.data, U̅.data, V̅.data, velocity_weight, tracer_weight,
                        dependencies=Event(device(arch)))
        wait(device(arch), event)
    end
end
