include(pwd() * "/src/Models/HydrostaticFreeSurfaceModels/split_explicit_free_surface.jl") # CHANGE TO USING MODULE EVENTUALLY
# TODO: clean up test, change to use interior
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
sefs = SplitExplicitForcing(grid, arch)
sefs = SplitExplicitFreeSurface(grid, arch)

sefs.Gᵁ
sefs.η .= 0.0
sefs.state.η === sefs.η
sefs.forcing.Gᵁ === sefs.Gᵁ

#=
∂t(η) = -∇⋅U⃗ 
∂t(U⃗) = - ∇η + f⃗
=#

@kernel function free_surface_substep_kernel_1!(grid, Δτ, η, U, V, Gᵁ, Gⱽ)
    i, j = @index(Global, NTuple)
    # ∂τ(U⃗) = - ∇η + G⃗
    @inbounds U[i, j, 1] +=  Δτ * (-∂xᶠᶜᵃ(i, j, 1,  grid, η) + Gᵁ[i, j, 1])
    @inbounds V[i, j, 1] +=  Δτ * (-∂yᶜᶠᵃ(i, j, 1,  grid, η) + Gⱽ[i, j, 1])
end

@kernel function free_surface_substep_kernel_2!(grid, Δτ, η, U, V, η̅, U̅, V̅, velocity_weight, tracer_weight)
    i, j = @index(Global, NTuple)
    # ∂τ(U⃗) = - ∇η + G⃗
    @inbounds η[i, j, 1] -=  Δτ * div_xyᶜᶜᵃ(i, j, 1, grid, U, V)
    # time-averaging
    @inbounds U̅[i, j, 1] +=  velocity_weight * U[i, j, 1]
    @inbounds V̅[i, j, 1] +=  velocity_weight * V[i, j, 1]
    @inbounds η̅[i, j, 1] +=  tracer_weight   * η[i, j, 1]
    # println("i:", i, ", j:", j, ", ", U[i,j])
end

function free_surface_substep!(arch, grid, Δτ, η, U, V, Gᵁ, Gⱽ, η̅, U̅, V̅, velocity_weight, tracer_weight)
    event = launch!(arch, grid, :xy, free_surface_substep_kernel_1!, 
            grid, Δτ, η, U, V, Gᵁ, Gⱽ,
            dependencies=Event(device(arch)))
    wait(event)
    
    event = launch!(arch, grid, :xy, free_surface_substep_kernel_2!, 
            grid, Δτ, η, U, V, η̅, U̅, V̅, velocity_weight, tracer_weight,
            dependencies=Event(device(arch)))
    wait(event)
            
end

##
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
##
# Test 2: Testing analytic solution 
U, V, η̅, U̅, V̅, Gᵁ, Gⱽ  = sefs.U, sefs.V, sefs.η̅, sefs.U̅, sefs.V̅, sefs.Gᵁ, sefs.Gⱽ
η = sefs.η
velocity_weight = 0.0
tracer_weight = 0.0

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
    fill_halo_regions!(η, arch)
    fill_halo_regions!(U, arch)
    fill_halo_regions!(V, arch)
    free_surface_substep!(arch, grid, Δτ, η, U, V, Gᵁ, Gⱽ, η̅, U̅, V̅, velocity_weight, tracer_weight)
end
# + correction for exact time
fill_halo_regions!(η, arch)
fill_halo_regions!(U, arch)
fill_halo_regions!(V, arch)
free_surface_substep!(arch, grid, Δτ_end, η, U, V, Gᵁ, Gⱽ, η̅, U̅, V̅, velocity_weight, tracer_weight)

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
U, V, η̅, U̅, V̅, Gᵁ, Gⱽ  = sefs.U, sefs.V, sefs.η̅, sefs.U̅, sefs.V̅, sefs.Gᵁ, sefs.Gⱽ
η = sefs.η

# set!(η, f(x,y)) k^2 = ω^2
kx = 2
ky = 3
gu_c = 1.0 
gv_c = 2.0 
η₀(x,y) = sin(kx * x) * sin(ky * y)
set!(η, η₀)

ω = sqrt(kx^2 + ky^2)
T = 2π/ω / 3 * 2
Δτ = 2π / maximum([Nx, Ny]) * 1e-2 # the last factor is essentially the order of accuracy
Nt = floor(Int, T/Δτ)
Δτ_end = T - Nt * Δτ

U  .= 0.0 # so that ∂ᵗη(t=0) = 0.0 
V  .= 0.0 # so that ∂ᵗη(t=0) = 0.0
η̅  .= 0.0
U̅  .= 0.0 
V̅  .= 0.0
Gᵁ .= gu_c
Gⱽ .= gv_c 
velocity_weights = ones(Nt+1) ./ Nt   # since taking Nt+1 timesteps
tracer_weights   = ones(Nt+1) ./ Nt   # since taking Nt+1 timesteps
velocity_weights[Nt+1] = Δτ_end / T   # since last timestep is different
tracer_weights[Nt+1] = Δτ_end / T     # since last timestep is different

print("The full timestep loop takes ")
tic = Base.time()

for i in 1:Nt
    velocity_weight = velocity_weights[i]
    tracer_weight = tracer_weights[i]
    fill_halo_regions!(η, arch)
    fill_halo_regions!(U, arch)
    fill_halo_regions!(V, arch)
    free_surface_substep!(arch, grid, Δτ, η, U, V, Gᵁ, Gⱽ, η̅, U̅, V̅, velocity_weight, tracer_weight)
end
# + correction for exact time
velocity_weight = velocity_weights[Nt+1]
tracer_weight   =   tracer_weights[Nt+1]
fill_halo_regions!(η, arch)
fill_halo_regions!(U, arch)
fill_halo_regions!(V, arch)
free_surface_substep!(arch, grid, Δτ_end, η, U, V, Gᵁ, Gⱽ, η̅, U̅, V̅, velocity_weight, tracer_weight)

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