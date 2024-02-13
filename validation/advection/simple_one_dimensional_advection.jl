using Oceananigans
using Oceananigans.Advection: MPData
using Oceananigans.Models.ShallowWaterModels: ConservativeFormulation
using JLD2
using OffsetArrays
using LinearAlgebra
using Test
using Adapt 

"""
This simulation is a simple 1D advection to test the 
validity of the stretched WENO scheme
"""

arch = CPU()

#parameters
N = 100

# regular "stretched" grid
Freg = range(0, 1, length = N+1)

# center-coarsened grid
Δstr(i, N) = i < N/4 ? 1 : (i > N*0.75 ? 1 : (i < N/2 ? 1.08 * (i - N/4) + 1 : 1.08 * (3*N/4 - i) + 1))
Fstr = zeros(Float64, N+1)
for i = 2:N+1
     Fstr[i] = Fstr[i-1] + Δstr(i-1, N)
end
Fstr ./= Fstr[end]

Freg = vcat(reverse(-Freg)[1:end-1], Freg)
Fstr = vcat(reverse(-Fstr)[1:end-1], Fstr)

Nx = 2N

solution = Dict()

# 1D grid constructions
grid = RectilinearGrid(arch, size = (Nx, 1), x = Freg,  y = (0, 1), halo = (7, 7), topology = (Periodic, Periodic, Flat))

# the initial condition
@inline G(x, β, z) = exp(-β*(x - z)^2)
@inline F(x, α, a) = √(max(1 - α^2*(x-a)^2, 0.0))

Z = -0.7
δ = 0.005
β = log(2)/(36*δ^2)
a = 0.5
α = 10

@inline function c₀_1D(x, y) 
    if x <= -0.6 && x >= -0.8
        return 1/6*(G(x, β, Z-δ) + 4*G(x, β, Z) + G(x, β, Z+δ))
    elseif x <= -0.2 && x >= -0.4
        return 1.0
    elseif x <= 0.2 && x >= 0.0
        return 1.0 - abs(10 * (x - 0.1))
    elseif x <= 0.6 && x >= 0.4
        return 1/6*(F(x, α, a-δ) + 4*F(x, α, a) + F(x, α, a+δ))
    else
        return 0.0
    end
end

@info "testing grids"

Δt_max   = 0.2 * minimum_xspacing(grid)
end_time = 2.0

@show tot_iter = end_time ÷ Δt_max
end_iter = tot_iter÷10

c_real = CenterField(grid)
formulation = ConservativeFormulation()
set!(c_real, c₀_1D)

schemes = [Centered(; order = 4), UpwindBiased(; order = 1), UpwindBiased(; order = 3), MPData(grid; iterations = 3), MPData(grid)]
names   = [:C, :U1, :U3, :M3, :MI]

for (tracer_advection, name) in zip(schemes, names)
    @info "Scheme $(summary(tracer_advection))"
    model = HydrostaticFreeSurfaceModel(; grid = grid,
                                       tracers = :c,
                                    velocities = PrescribedVelocityFields(; u = (x, y, z) -> 1, v = (x, y, z) -> 0, w = (x, y, z) -> 0),
                                      buoyancy = nothing,
                              tracer_advection)
          
    set!(model, c=c₀_1D)
    c  = model.tracers.c

    for i = 1:end_iter
        for j = 1:10
            time_step!(model, Δt_max)
        end
        csim  = Array(interior(c, :, 1, 1))
        solution[(name, Int(i))] = csim
    end
end
