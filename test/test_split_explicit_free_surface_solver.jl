include(pwd() * "/src/Models/HydrostaticFreeSurfaceModels/split_explicit_free_surface.jl") # CHANGE TO USING MODULE EVENTUALLY
include(pwd() * "/src/Models/HydrostaticFreeSurfaceModels/split_explicit_free_surface_kernels.jl")
# TODO: clean up test, change to use interior
# TODO: clean up substep function so that it only takes in SplitExplicitFreeSurface
using Oceananigans.Utils
using Oceananigans.BoundaryConditions
using Oceananigans.Operators
using KernelAbstractions

const g_Earth = 9.80665

arch = Oceananigans.GPU()
FT = Float64
topology = (Periodic, Periodic, Bounded)
Nx = Ny = Nz = 16 * 8 
Nx = 128
Ny = 64
Lx = Ly = Lz = 2π
grid = RegularRectilinearGrid(topology=topology, size=(Nx, Ny, Nz), x=(0, Lx), y=(0, Ly), z=(-Lz, 0))

tmp = SplitExplicitFreeSurface()
sefs = SplitExplicitState(grid, arch)
sefs = SplitExplicitAuxiliary(grid, arch)
sefs = SplitExplicitFreeSurface(grid, arch)

sefs.Gᵁ
sefs.η .= 0.0
sefs.state.η === sefs.η
sefs.auxiliary.Gᵁ === sefs.Gᵁ

##
# Test 1: Evaluating the RHS with a simple test
U, V, η̅, U̅, V̅, Gᵁ, Gⱽ  = sefs.U, sefs.V, sefs.η̅, sefs.U̅, sefs.V̅, sefs.Gᵁ, sefs.Gⱽ
Hᶠᶜ, Hᶜᶠ = sefs.Hᶠᶜ, sefs.Hᶜᶠ
Hᶠᶜ .= 1/sefs.parameters.g
Hᶜᶠ .= 1/sefs.parameters.g
η = sefs.η
velocity_weight = 0.0
free_surface_weight = 0.0
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

free_surface_substep!(arch, grid, Δτ, sefs, 1)

U_computed = Array(U.data.parent)[2:Nx+1, 2:Ny+1]

U_exact = (reshape(-cos.(grid.xF), (length(grid.xC), 1)) .+ reshape(0 * grid.yC, (1, length(grid.yC))))[2:Nx+1, 2:Ny+1]

println("maximum error is ", maximum(abs.(U_exact - U_computed)))
##
# Test 2: Testing analytic solution 
U, V, η̅, U̅, V̅, Gᵁ, Gⱽ  = sefs.U, sefs.V, sefs.η̅, sefs.U̅, sefs.V̅, sefs.Gᵁ, sefs.Gⱽ
sefs.Hᶠᶜ .= 1/sefs.parameters.g
sefs.Hᶜᶠ .= 1/sefs.parameters.g
η = sefs.η
velocity_weight = 0.0
free_surface_weight = 0.0

T = 2π
Δτ = 2π / maximum([Nx, Ny]) * 5e-2 # the last factor is essentially the order of accuracy
Nt = floor(Int, T/Δτ)
Δτ_end = T - Nt * Δτ

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

print("The full timestep loop takes ")
tic = Base.time()
for i in 1:Nt
    free_surface_substep!(arch, grid, Δτ, sefs, 1)
end
# + correction for exact time
free_surface_substep!(arch, grid, Δτ_end, sefs, 1)

toc = Base.time()
println(toc - tic, " seconds")

U_computed = Array(U.data.parent)[2:Nx+1, 2:Ny+1]
η_computed = Array(η.data.parent)[2:Nx+1, 2:Ny+1]
set!(η, η₀)
set!(U, U₀)
U_exact = Array(U.data.parent)[2:Nx+1, 2:Ny+1]
η_exact = Array(η.data.parent)[2:Nx+1, 2:Ny+1]

err1 = maximum(abs.(U_computed - U_exact))
err2 = maximum(abs.(η_computed - η_exact))

println("The U error is ", err1)
println("The η error is ", err2)


println("The L∞ norm of U is ", maximum(abs.(U_computed)))
println("The L∞ norm of η is ", maximum(abs.(η_computed)))

##
# Test 3: Testing analytic solution to 
# ∂ₜη + ∇⋅U⃗ = 0
# ∂ₜU⃗ + ∇η  = G⃗
kx = 2
ky = 3
ω = sqrt(kx^2 + ky^2)
T = 2π/ω / 3 * 2
Δτ = 2π / maximum([Nx, Ny]) * 1e-2 # error mostly spatially dependent, except in the averaging
Nt = floor(Int, T/Δτ)
Δτ_end = T - Nt * Δτ


sefs = SplitExplicitFreeSurface(grid, arch, substeps = Nt+1)
U, V, η̅, U̅, V̅, Gᵁ, Gⱽ  = sefs.U, sefs.V, sefs.η̅, sefs.U̅, sefs.V̅, sefs.Gᵁ, sefs.Gⱽ
η = sefs.η
sefs.Hᶠᶜ .= 1/sefs.parameters.g # to make life easy
sefs.Hᶜᶠ .= 1/sefs.parameters.g # to make life easy

# set!(η, f(x,y)) k^2 = ω^2
gu_c = 1.0 
gv_c = 2.0 
η₀(x,y) = sin(kx * x) * sin(ky * y)
set!(η, η₀)



U  .= 0.0 # so that ∂ᵗη(t=0) = 0.0 
V  .= 0.0 # so that ∂ᵗη(t=0) = 0.0
η̅  .= 0.0
U̅  .= 0.0 
V̅  .= 0.0
Gᵁ .= gu_c
Gⱽ .= gv_c 
# overwrite weights
sefs.velocity_weights .= ones(Nt+1) ./ Nt        # since taking Nt+1 timesteps
sefs.free_surface_weights   .= ones(Nt+1) ./ Nt  # since taking Nt+1 timesteps
sefs.velocity_weights[Nt+1] = Δτ_end / T         # since last timestep is different
sefs.free_surface_weights[Nt+1] = Δτ_end / T     # since last timestep is different

print("The full timestep loop takes ")
tic = Base.time()

for i in 1:Nt
    free_surface_substep!(arch, grid, Δτ, sefs, i)
end
# + correction for exact time
free_surface_substep!(arch, grid, Δτ_end, sefs, Nt+1)

toc = Base.time()
println(toc - tic, " seconds")

η_computed = Array(η.data.parent)[2:Nx+1, 2:Ny+1]
U_computed = Array(U.data.parent)[2:Nx+1, 2:Ny+1]
V_computed = Array(V.data.parent)[2:Nx+1, 2:Ny+1]

η̅_computed = Array(η̅.data.parent)[2:Nx+1, 2:Ny+1]
U̅_computed = Array(U̅.data.parent)[2:Nx+1, 2:Ny+1]
V̅_computed = Array(V̅.data.parent)[2:Nx+1, 2:Ny+1]

set!(η, η₀)
# ∂ₜₜ(η) = Δη
η_exact = cos(ω * T ) * Array(η.data.parent)[2:Nx+1, 2:Ny+1]

U₀(x,y) =  kx * cos(kx * x) * sin(ky * y) # ∂ₜU = - ∂x(η), since we know η
set!(U, U₀)
U_exact =  -(sin(ω * T) * 1 /ω)  .* Array(U.data.parent)[2:Nx+1, 2:Ny+1] .+ gu_c * T

V₀(x,y) =  ky * sin(kx * x) * cos(ky * y) # ∂ₜV = - ∂y(η), since we know η
set!(V, V₀)
V_exact =  -(sin(ω * T) * 1 /ω)  .* Array(V.data.parent)[2:Nx+1, 2:Ny+1] .+ gv_c * T

η̅_exact = (sin(ω * T)/ω - sin(ω * 0)/ω)/T * Array(η.data.parent)[2:Nx+1, 2:Ny+1]
U̅_exact = (cos(ω * T) * 1 /ω^2 - cos(ω * 0) * 1 /ω^2)/T * Array(U.data.parent)[2:Nx+1, 2:Ny+1] .+ gu_c * T/2
V̅_exact = (cos(ω * T) * 1 /ω^2 - cos(ω * 0) * 1 /ω^2)/T * Array(V.data.parent)[2:Nx+1, 2:Ny+1] .+ gv_c * T/2

errU = maximum(abs.(U_computed - U_exact)) / maximum(abs.(U_exact)) 
errV = maximum(abs.(V_computed - V_exact)) / maximum(abs.(V_exact)) 
errη = maximum(abs.(η_computed - η_exact)) / maximum(abs.(η_exact)) 

errU̅ = maximum(abs.(U̅_computed - U̅_exact)) 
errV̅ = maximum(abs.(V̅_computed - V̅_exact)) 
errη̅ = maximum(abs.(η̅_computed - η̅_exact)) 

println("The relative U L∞ error is ", errU)
println("The relative V L∞ error is ", errV)
println("The relative η L∞ error is ", errη)

println("The U̅ L∞ error is ", errU̅)
println("The V̅ L∞ error is ", errV̅)
println("The η̅ L∞ error is ", errη̅)

println("The L∞ norm of U is ", maximum(abs.(U_computed)))
println("The L∞ norm of V is ", maximum(abs.(V_computed)))
println("The L∞ norm of η is ", maximum(abs.(η_computed)))

println("The L∞ norm of U̅ is ", maximum(abs.(U̅_computed)))
println("The L∞ norm of V̅ is ", maximum(abs.(V̅_computed)))
println("The L∞ norm of η̅ is ", maximum(abs.(η̅_computed)))