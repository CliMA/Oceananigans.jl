using Oceananigans
using Oceananigans.Grids: min_Δx
using JLD2
using OffsetArrays
using BenchmarkTools
using Test
using LinearAlgebra
using Plots
"""

This simulation is a simple 1D advection of a gaussian function, to test the 
validity of the stretched WENO scheme
    
"""
Nx   = 20
arch = CPU()

# regular "stretched" mesh
xF_reg = range(0,1,length = Nx+1)

#stretched mesh
Δx_str = zeros(Nx); Δx_str .= 100
Δx_str[4] = 65; Δx_str[5] = 12; Δx_str[6] = 48; Δx_str[7] = 208; Δx_str[8] = 15

xF_str = zeros(Float64, Nx+1)
xF_str[1] = 0
for i in 2:Nx+1
    xF_str[i] = xF_str[i-1] + Δx_str[i-1]
end
xF_str = xF_str ./ xF_str[end]
   
function Δx_str2(i, N)
    if i < N/4
     return 1
    elseif i > N/4*3
     return 1
    elseif i<N/2
     return 1.2 * (i - N/4) + 1
    else
     return  1.2 * (3*N/4 - i) + 1
    end
 end   

 xF = zeros(Float64, Nx+1)

 for i = 2:Nx+1
     xF[i] = xF[i-1] + Δx_str2(i-1, Nx)
 end

 xF ./= xF[end]

solution  = Dict()
real_sol  = Dict()
coord     = Dict()
residual  = Dict()

grid_reg  = RectilinearGrid(size = (Nx,), x = xF_reg, halo = (3,), topology = (Periodic, Flat, Flat), architecture = arch)    
grid_str  = RectilinearGrid(size = (Nx,), x = xF_str, halo = (3,), topology = (Periodic, Flat, Flat), architecture = arch)    
grid_str2 = RectilinearGrid(size = (Nx,), x = xF    , halo = (3,), topology = (Periodic, Flat, Flat), architecture = arch)    

advection = [CenteredSecondOrder(), CenteredFourthOrder(), WENO5(), WENO5()]

# Checking the accuracy of different schemes with different settings

for grid in [grid_reg, grid_str, grid_str2], (adv, scheme) in enumerate(advection) 


    grid == grid_reg ? gr = :reg : grid == grid_str ? gr = :str : gr = :str2

    U = Field(Face, Center, Center, arch, grid)
    parent(U) .= 1 

    if adv == 4
        scheme = WENO5(grid = grid)
    end

    model = HydrostaticFreeSurfaceModel(architecture = arch,
                                                grid = grid,
                                             tracers = :c,
                                    tracer_advection = scheme,
                                          velocities = PrescribedVelocityFields(u=U), 
                                            coriolis = nothing,
                                             closure = nothing,
                                            buoyancy = nothing)

    
    Δt_max   = 0.5 * min_Δx(grid)
    end_time = 1000 * Δt_max
    c₀(x, y, z) = 10*exp(-((x-0.5)/0.1)^2)
                                            
    set!(model, c=c₀)
    c = model.tracers.c

    x        = grid.xᶜᵃᵃ[1:grid.Nx]
    creal    = Array(c₀.(mod.((x .- end_time),1), 0, 0))

    simulation = Simulation(model,
                            Δt = Δt_max,
                            stop_time = end_time)                           
    run!(simulation, pickup=false)

    ctest   = Array(parent(c.data))
    offsets = (c.data.offsets[1],  c.data.offsets[2],  c.data.offsets[3])
    ctemp   = OffsetArray(ctest, offsets)
    ctests  = ctemp[1:grid.Nx, 1, 1]

    real_sol[(gr)] = creal
    coord[(gr)]    = x
    
    residual[(adv, gr)] = norm(abs.(creal .- ctests))
    solution[(adv, gr)] = ctests
end

"""
Now test a 2D simulation (to do)
"""


