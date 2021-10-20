include(pwd() * "/src/Models/HydrostaticFreeSurfaceModels/split_explicit_free_surface.jl")
const g_Earth = 9.80665

arch = Oceananigans.GPU()
FT = Float64
topology = (Periodic, Periodic, Bounded)
Nx = Ny = Nz = 16*4*4
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
using Oceananigans.BoundaryConditions
using Oceananigans.Operators
using KernelAbstractions

# TODO: add g * H[i,j]
@kernel function free_surface_substep_kernel!(grid, Δτ, η, U, V, Gᵁ, Gⱽ, η̅, U̅, V̅, velocity_weight, tracer_weight)
    i, j = @index(Global, NTuple)
    # ∂τ(U⃗) = - ∇η + G⃗
    @inbounds U[i, j, 1] +=  Δτ * (-∂xᶠᶜᵃ(i, j, 1,  grid, η) + Gᵁ[i, j, 1])
    @inbounds V[i, j, 1] +=  Δτ * (-∂yᶜᶠᵃ(i, j, 1,  grid, η) + Gⱽ[i, j, 1])
    @synchronize # since ∇⋅U⃗ uses nonlocal threads
    # ∂τ(η) = - ∇⋅U⃗
    @inbounds η[i, j, 1] -=  Δτ * div_xyᶜᶜᵃ(i, j, 1, grid, U, V)
    # time-averaging
    @inbounds U̅[i, j, 1] +=  velocity_weight * U[i, j, 1]
    @inbounds V̅[i, j, 1] +=  velocity_weight * V[i, j, 1]
    @inbounds η̅[i, j, 1] +=  tracer_weight   * η[i, j, 1]
end

function free_surface_substep!(arch, grid, Δτ, η, U, V, Gᵁ, Gⱽ, η̅, U̅, V̅, velocity_weight, tracer_weight)
    launch!(arch, grid, :xy, free_surface_substep_kernel!, 
                        grid, Δτ, 
                        η.data, U.data, V.data, Gᵁ.data, Gⱽ.data, 
                        η̅.data, U̅.data, V̅.data, 
                        velocity_weight, tracer_weight,
                        dependencies=Event(device(arch)))
end

# Test 1: Evaluating the RHS with a simple test
U, V, η̅, U̅, V̅, Gᵁ, Gⱽ  = sefs.U, sefs.V, sefs.η̅, sefs.U̅, sefs.V̅, sefs.Gᵁ, sefs.Gⱽ
η = sefs.η
velocity_weight = 0.0
tracer_weight = 0.0
Δτ = 1.0

# set!(η, f(x,y))
η₀(x,y) = sin(x)
set!(η, η₀)
U₀(x,y) = 0.0
set!(U, U₀)
V₀(x,y) = 0.0
set!(V, V₀)

η̅  .= 0.0
U̅  .= 0.0 
V̅  .= 0.0
Gᵁ .= 0.0
Gⱽ .= 0.0 

fill_halo_regions!(η, arch)
fill_halo_regions!(U, arch)
fill_halo_regions!(V, arch)

free_surface_substep!(arch, grid, Δτ, η, U, V, Gᵁ, Gⱽ, η̅, U̅, V̅, velocity_weight, tracer_weight)

U_computed = Array(U.data.parent)[2:Nx+1, 2:Ny+1]

U_exact = (reshape(-cos.(grid.xF), (length(grid.xC), 1)) .+ reshape(0 * grid.yC, (1, length(grid.yC))))[2:Nx+1, 2:Ny+1]

println("maximum error is ", maximum(abs.(U_exact - U_computed)))

# Test 2: Testing analytic solution 
U, V, η̅, U̅, V̅, Gᵁ, Gⱽ  = sefs.U, sefs.V, sefs.η̅, sefs.U̅, sefs.V̅, sefs.Gᵁ, sefs.Gⱽ
η = sefs.η
velocity_weight = 0.0
tracer_weight = 0.0

T = 2π
Δτ = 2π / sqrt(Nx^2 + Ny^2) * 1e-2 # the last factor is essentially the order of accuracy
Nt = floor(Int, T/Δτ)

# set!(η, f(x,y))
η₀(x,y) = sin(x)
set!(η, η₀)
U₀(x,y) = cos(x)
set!(U, U₀)
V₀(x,y) = 0.0
set!(V, V₀)

η̅  .= 0.0
U̅  .= 0.0 
V̅  .= 0.0
Gᵁ .= 0.0
Gⱽ .= 0.0 

for i in 1:Nt
    fill_halo_regions!(η, arch)
    fill_halo_regions!(U, arch)
    fill_halo_regions!(V, arch)
    free_surface_substep!(arch, grid, Δτ, η, U, V, Gᵁ, Gⱽ, η̅, U̅, V̅, velocity_weight, tracer_weight)
end

U_computed = Array(U.data.parent)[2:Nx+1, 2:Ny+1]
η_computed = Array(η.data.parent)[2:Nx+1, 2:Ny+1]
set!(η, η₀)
set!(U, U₀)
U_exact = Array(U.data.parent)[2:Nx+1, 2:Ny+1]
η_exact = Array(η.data.parent)[2:Nx+1, 2:Ny+1]

err1 = maximum(abs.(U_computed - U_exact))
err2 = maximum(abs.(η_computed - η_exact))

println("The first error is ", err1)
println("The second error is ", err2)




function substep!(arch, grid, Δt, split_explicit_free_surface::SplitExplicitFreeSurface)
    # unpack
    sefs = split_explicit_free_surface
    η, U, V, η̅, U̅, V̅, = sefs.U, sefs.V, sefs.η̅, sefs.U̅, sefs.V̅, sefs.U, sefs.V
    Gᵁ, Gⱽ = sefs.Gᵁ, sefs.Gⱽ
    substeps = sefs.settings.substeps
    velocity_weights = sefs.settings.velocity_weights
    tracer_weights = sefs.settings.tracer_weights

    # TODO: DEFINE Δτ appropriately
    Δτ = 2 * Δt / substeps # go twice as far for averaging

    for i in 1:substeps
        velocity_weight = velocity_weights[i]
        tracer_weight = tracer_weights[i]
        # fill_halo_regions! is blocking 
        fill_halo_regions!(η, arch)
        fill_halo_regions!(U, arch)
        fill_halo_regions!(V, arch)
        # substep 
        event = launch!(arch, grid, :xy, free_surface_substep_kernel!, 
                        grid, Δτ, 
                        η.data, U.data, V.data, Gᵁ.data, Gⱽ.data, 
                        η̅.data, U̅.data, V̅.data, 
                        velocity_weight, tracer_weight,
                        dependencies=Event(device(arch)))
        wait(device(arch), event)
    end
end
