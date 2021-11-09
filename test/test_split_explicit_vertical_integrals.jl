using Revise

include(pwd() * "/src/Models/HydrostaticFreeSurfaceModels/split_explicit_free_surface.jl")
include(pwd() * "/src/Models/HydrostaticFreeSurfaceModels/split_explicit_free_surface_kernels.jl")

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
