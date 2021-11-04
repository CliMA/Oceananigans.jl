include(pwd() * "/src/Models/HydrostaticFreeSurfaceModels/split_explicit_free_surface.jl")

using Revise
using Oceananigans.Utils
using Oceananigans.BoundaryConditions
using Oceananigans.Operators
using KernelAbstractions
using KernelAbstractions.Extras.LoopInfo: @unroll

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

@kernel function barotropic_corrector_kernel!(u, v, U̅, V̅, U, V, Hᶠᶜ, Hᶜᶠ)
    i, j, k = @index(Global, NTuple)
    u[i,j,k] = u[i,j,k] + (-U[i,j] + U̅[i,j] )/ Hᶠᶜ[i,j]
    v[i,j,k] = v[i,j,k] + (-V[i,j] + V̅[i,j] )/ Hᶜᶠ[i,j]
end

# may need to do Val(Nk) since it may not be known at compile. Also figure out where to put H
function barotropic_corrector!(arch, grid, free_surface_state, u, v, Δz, Nk)
    sefs = free_surface_state
    U, V, U̅, V̅ = sefs.U, sefs.V, sefs.U̅, sefs.V̅
    Hᶠᶜ, Hᶜᶠ = sefs.Hᶠᶜ, sefs.Hᶜᶠ
    # take out "bad" barotropic mode, 
    # !!!! reusing U and V for this storage since last timestep doesn't matter
    naive_barotropic_mode!(arch, grid, U, u, Δz, Nk)
    naive_barotropic_mode!(arch, grid, V, v, Δz, Nk)
    # add in "good" barotropic mode
    
    event = launch!(arch, grid, :xyz, barotropic_corrector_kernel!, 
            u, v, U̅, V̅, U, V, Hᶠᶜ, Hᶜᶠ,
            dependencies=Event(device(arch)))
    wait(event)        
    
end

# Define CPU Test
const g_Earth = 9.80665

arch = Oceananigans.CPU()
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
sefs = SplitExplicitAuxiliary(grid, arch)
sefs = SplitExplicitFreeSurface(grid, arch)

U, V, η̅, U̅, V̅, Gᵁ, Gⱽ  = sefs.U, sefs.V, sefs.η̅, sefs.U̅, sefs.V̅, sefs.Gᵁ, sefs.Gⱽ

u = Field(Face, Center, Center, arch, grid)
v = Field(Center, Face, Center, arch, grid)

# Test 1: Check that averages have been set to zero on the cpu
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

# Test 2: Check that vertical integrals work on the CPU(). The following should be "inexact"
Δz = zeros(Nz)
Δz .= grid.Δz

set_u_check(x,y,z) =  cos( (π/2) * z / Lz)
set_U_check(x,y) = (sin(0) - (-2* Lz /(π)))
set!(u, set_u_check)
exact_U = copy(U)
set!(exact_U, set_U_check)
naive_barotropic_mode!(arch, grid, U, u, Δz, Nz)
tolerance = 1e-3
all((interior(U) .- interior(exact_U)) .< tolerance)

set_v_check(x,y,z) = sin(x*y) * cos( (π/2) * z / Lz)
set_V_check(x,y) = sin(x*y) * (sin(0) - (-2* Lz /(π)))
set!(v, set_v_check)
exact_V = copy(V)
set!(exact_V, set_V_check)
naive_barotropic_mode!(arch, grid, V, v, Δz, Nz)
all((interior(V) .- interior(exact_V)) .< tolerance)

# Test 3: Check that vertical integrals work on the CPU(). The following should be "exact"
Δz = zeros(Nz)
Δz .= grid.Δz

u .= 0.0
U .= 1.0
naive_barotropic_mode!(arch, grid, U, u, Δz, Nz)
all(U.data.parent .== 0.0)

u .= 1.0
U .= 1.0
naive_barotropic_mode!(arch, grid, U, u, Δz, Nz)
all(interior(U) .≈ Lz)

set_u_check(x,y,z) = sin(x)
set_U_check(x,y) = sin(x) * Lz
set!(u, set_u_check)
exact_U = copy(U)
set!(exact_U, set_U_check)
naive_barotropic_mode!(arch, grid, U, u, Δz, Nz)
all(interior(U) .≈ interior(exact_U))

set_v_check(x,y,z) = sin(x) * z * cos(y)
set_V_check(x,y) = -sin(x) * Lz^2/2.0 * cos(y)
set!(v, set_v_check)
exact_V = copy(V)
set!(exact_V, set_V_check)
naive_barotropic_mode!(arch, grid, V, v, Δz, Nz)
all(interior(V) .≈ interior(exact_V))

# Test 3: Check that vertical integrals work on the GPU(). Reuses the Arrays Defined in the CPU section
arch = Oceananigans.GPU()
FT = Float64
topology = (Periodic, Periodic, Bounded)
grid = RegularRectilinearGrid(topology=topology, size=(Nx, Ny, Nz), x=(0, Lx), y=(0, Ly), z=(-Lz, 0))
tmp = SplitExplicitFreeSurface()
sefs = SplitExplicitState(grid, arch)
sefs = SplitExplicitAuxiliary(grid, arch)
sefs = SplitExplicitFreeSurface(grid, arch)
U, V, η̅, U̅, V̅, Gᵁ, Gⱽ  = sefs.U, sefs.V, sefs.η̅, sefs.U̅, sefs.V̅, sefs.Gᵁ, sefs.Gⱽ
Δz = Oceananigans.CUDA.CuArray(Δz)

set_U_check(x,y) = sin(x) * Lz
naive_barotropic_mode!(arch, grid, U, Oceananigans.CUDA.CuArray(u), Δz, Nz)
all(Array(interior(U)) .≈ interior(exact_U))

naive_barotropic_mode!(arch, grid, V, Oceananigans.CUDA.CuArray(v), Δz, Nz)
all(Array(interior(V)) .≈ interior(exact_V))
##
# Test 4: Test Barotropic Correction
arch = Oceananigans.CPU()
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
sefs = SplitExplicitAuxiliary(grid, arch)
sefs = SplitExplicitFreeSurface(grid, arch)

U, V, η̅, U̅, V̅, Gᵁ, Gⱽ  = sefs.U, sefs.V, sefs.η̅, sefs.U̅, sefs.V̅, sefs.Gᵁ, sefs.Gⱽ

u = Field(Face, Center, Center, arch, grid)
v = Field(Center, Face, Center, arch, grid)
u_corrected = copy(u)
v_corrected = copy(v)

set_u(x,y,z) =  z+Lz/2 + sin(x)
set_U̅(x,y) = cos(x) * Lz
set_u_corrected(x,y,z) =  z+Lz/2 + cos(x)
set!(u, set_u)
set!(U̅, set_U̅)
set!(u_corrected, set_u_corrected)

set_v(x,y,z) =  (z+Lz/2)*sin(y) + sin(x)
set_V̅(x,y) = (cos(x) + x) * Lz
set_v_corrected(x,y,z) =  (z+Lz/2)*sin(y) + cos(x)+x
set!(v, set_v)
set!(V̅, set_V̅)
set!(v_corrected, set_v_corrected)

sefs.Hᶠᶜ .= Lz
sefs.Hᶜᶠ .= Lz

Δz = zeros(Nz)
Δz .= grid.Δz

barotropic_corrector!(arch, grid, sefs, u, v, Δz, Nz)
all((u .- u_corrected) .< 1e-14)
all((v .- v_corrected) .< 1e-14)
#=
function test_val(::Val{info}) where info
    return info.dim
end
proxy_info = (; dim=3,Nq=3)
test_val(Val(proxy_info))
=#
