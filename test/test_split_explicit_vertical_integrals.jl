include(pwd() * "/src/Models/HydrostaticFreeSurfaceModels/split_explicit_free_surface.jl")

using Revise
using Oceananigans.Utils
using Oceananigans.BoundaryConditions
using Oceananigans.Operators
using KernelAbstractions
using KernelAbstractions.Extras.LoopInfo: @unroll

const g_Earth = 9.80665

arch = Oceananigans.GPU()
FT = Float64
topology = (Periodic, Periodic, Bounded)
Nx = Ny = 16 * 8 
Nz = 32
Nx = 128 
Ny = 64 
Lx = Ly = Lz = 2π
grid = RegularRectilinearGrid(topology=topology, size=(Nx, Ny, Nz), x=(0, Lx), y=(0, Ly), z=(-Lz, 0))

tmp = SplitExplicitFreeSurface()
sefs = SplitExplicitState(grid, arch)
sefs = SplitExplicitForcing(grid, arch)
sefs = SplitExplicitFreeSurface(grid, arch)

U, V, η̅, U̅, V̅, Gᵁ, Gⱽ  = sefs.U, sefs.V, sefs.η̅, sefs.U̅, sefs.V̅, sefs.Gᵁ, sefs.Gⱽ

u = Field(Face, Center, Center, arch, grid)
v = Field(Center, Face, Center, arch, grid)

@kernel function set_average_zero_kernel!(η̅, U̅, V̅)
    i, j = @index(Global, NTuple)
    @inbounds U̅[i, j, 1] = 0.0
    @inbounds V̅[i, j, 1] = 0.0
    @inbounds η̅[i, j, 1] = 0.0
end

function set_average_to_zero!(arch, grid, η̅, U̅, V̅)
    event = launch!(arch, grid, :xy, set_average_zero_kernel!, 
            η̅, U̅, V̅,
            dependencies=Event(device(arch)))
    wait(event)        
end

@kernel function barotropic_mode_kernel!(U, V, Gᵁ, Gⱽ, u, v, gᵘ, gᵛ, Δz, Nk)
    i, j = @index(Global, NTuple)

    kk = 1
    @inbounds U[i, j, 1] = Δz[kk] * u[i,j,kk]
    @inbounds V[i, j, 1] = Δz[kk] * v[i,j,kk]

    @inbounds Gᵁ[i, j, 1] = Δz[kk] * gᵘ[i,j,kk]
    @inbounds Gⱽ[i, j, 1] = Δz[kk] * gᵛ[i,j,kk]

    @unroll for k in 2:Nk
        @inbounds U[i, j, 1] += Δz[k] * u[i,j,k]
        @inbounds V[i, j, 1] += Δz[k] * v[i,j,k]

        @inbounds Gᵁ[i, j, 1] += Δz[k] * gᵘ[i,j,k]
        @inbounds Gⱽ[i, j, 1] += Δz[k] * gᵛ[i,j,k]
    end
end

# may need to do Val(Nk) since it may not be known at compile
function barotropic_mode!(arch, grid, U, V, Gᵁ, Gⱽ, u, v, gᵘ, gᵛ, Δz, Nk)
    event = launch!(arch, grid, :xy, barotropic_mode_kernel!, 
            U, V, Gᵁ, Gⱽ, u, v, gᵘ, gᵛ, Δz, Nk,
            dependencies=Event(device(arch)))
    wait(event)        
end

@kernel function naive_barotropic_mode_kernel!(ϕ̅, ϕ, Δz, Nk)
    i, j = @index(Global, NTuple)
    # hand unroll first loop 
    @inbounds ϕ̅[i, j, 1] = Δz[1] * ϕ[i,j,1]
    for k in 2:Nk
        @inbounds ϕ̅[i, j, 1] += Δz[k] * ϕ[i,j,k] 
    end
end

# may need to do Val(Nk) since it may not be known at compile
function naive_barotropic_mode!(arch, grid, ϕ̅, ϕ, Δz, Nk)
    event = launch!(arch, grid, :xy, naive_barotropic_mode_kernel!, 
            ϕ̅, ϕ, Δz, Nk,
            dependencies=Event(device(arch)))
    wait(event)        
end

# Test 1: Check that averages have been set to zero
# set equal to something else
η̅ .= U̅ .= V̅ .= 1.0
# now set equal to zero
set_average_to_zero!(arch, grid, η̅, U̅, V̅)
# don't forget the ghost points
fill_halo_regions!(η̅, arch)
fill_halo_regions!(U̅, arch)
fill_halo_regions!(V̅, arch)
# check
all(η̅.data.parent .== 0.0)
all(U̅.data.parent .== 0.0)
all(V̅.data.parent .== 0.0)

# Test 2: Check that vertical integrals work
Δz = zeros(Nz)
Δz .= grid.Δz
if arch == Oceananigans.GPU()
    Δz = Oceananigans.CUDA.CuArray(Δz)
end

u .= 0.0
U .= 1.0
naive_barotropic_mode!(arch, grid, U, u, Δz, Nz)
all(U.data.parent .== 0.0)

u .= 1.0
U .= 1.0
naive_barotropic_mode!(arch, grid, U, u, Δz, Nz)
all(interior(U) .≈ Lz)

