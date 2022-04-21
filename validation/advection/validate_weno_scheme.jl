using Oceananigans
using Oceananigans.Grids: min_Δx, min_Δy, min_Δz
using JLD2
using OffsetArrays
using BenchmarkTools
using LinearAlgebra
using Plots
using Test
using Adapt 

"""
This simulation is a simple 1D advection to test the 
validity of the stretched WENO scheme
"""

function multiple_steps!(model)
    for i = 1:1000
        time_step!(model, 1e-6)
    end
    return nothing
end

#parameters
N    = 40
arch = CPU()
iter = 1000

# regular "stretched" grid
Freg = range(0, 1, length = N+1)

# alternating grid 
Fsaw(j) = 1 / N  * (j - 1) + 0.1 * 1 / N * mod(j - 1, 2)

# center-coarsened grid
Δstr2(i, N) = i < N/4 ? 1 : ( i > N*0.75 ? 1 : ( i < N/2 ? 1.2 * (i - N/4) + 1 : 1.2 * (3*N/4 - i) + 1  )  ) 
Fstr2 = zeros(Float64, N+1)
for i = 2:N+1
     Fstr2[i] = Fstr2[i-1] + Δstr2(i-1, N)
end
Fstr2 ./= Fstr2[end]

# solutions
solution  = Dict()
real_sol  = Dict()
coord     = Dict()
time      = Dict()

# 1D grid constructions
grid_reg  = RectilinearGrid(arch, size = N, x = Freg,  halo = 3, topology = (Periodic, Flat, Flat))    
grid_str  = RectilinearGrid(arch, size = N, x = Fsaw,  halo = 3, topology = (Periodic, Flat, Flat))    
grid_str2 = RectilinearGrid(arch, size = N, x = Fstr2, halo = 3, topology = (Periodic, Flat, Flat))    

# placeholder for the four different advection schemes 
#  (1) WENO5(), 
#  (2) WENO5(grid=grid),
#  (3) WENO5(grid=grid, stretched_smoothness=true),
#  (4) WENO5(grid=grid, stretched_smoothness=true, zweno=true)

advection = [WENO5(), WENO5(), WENO5(), WENO5()]
schemes   = [:wreg, :wstr, :wstrS, :wstrZ]

# mask for the initial condition
mask(y) = (y < 0.6 && y > 0.4) ? 1 : 0
c₀_1D(x, y, z) = @. 10 * exp(-(x - 0.5)^2 / 0.2^2)
c₀_2D(x, y, z) = mask(x) * mask(y) 

# Checking the accuracy of different schemes with different settings
for (gr, grid) in enumerate([grid_reg, grid_str, grid_str2])
    
    U = Field{Face, Center, Center}(grid)
    parent(U) .= 1

    Δt_max   = 0.2 * min_Δx(grid)
    end_time = iter * Δt_max
                                            
    for (adv, scheme) in enumerate(advection) 
        if adv == 2
            scheme = WENO5(grid = grid)
        end
        if adv == 3
            scheme = WENO5(grid = grid, stretched_smoothness = true)
        end
        if adv == 4
            scheme = WENO5(grid = grid, stretched_smoothness = true, zweno = true)
        end

        model = HydrostaticFreeSurfaceModel(        grid = grid,
                                                 tracers = :c,
                                        tracer_advection = scheme,
                                              velocities = PrescribedVelocityFields(u=U), 
                                                coriolis = nothing,
                                                 closure = nothing,
                                                buoyancy = nothing)
        
        set!(model, c=c₀_1D)
        c = model.tracers.c

        simulation = Simulation(model,
                                Δt = Δt_max,
                                stop_time = end_time)
        
        
        for i = 1:end_time/Δt_max
            csim                             = adapt(CPU(), c)
            solution[(schemes[adv], Int(i))] = csim[1:N, 1, 1]
            
            for j = 1:10
                time_step!(model, Δt_max)
            end
        end
        
        time[(schemes[adv], gr)] = @belapsed multiple_steps!($model)
    end
    
    x = adapt(CPU(), grid.xᶜᵃᵃ)
    x = x[1:N]

    anim = @animate for i ∈ 1:end_time/Δt_max
        plot(x, c₀_1D(x, 1, 1), seriestype=:scatter, legend = false, title ="red: U, blue: S, green: β, black: Z") 
        plot!(x, solution[(schemes[1], Int(i))], linewidth = 4, linecolor =:red  , legend = false) 
        plot!(x, solution[(schemes[2], Int(i))], linewidth = 3, linecolor =:blue , legend = false) 
        plot!(x, solution[(schemes[3], Int(i))], linewidth = 2, linecolor =:green, legend = false)
        plot!(x, solution[(schemes[4], Int(i))], linewidth = 1, linecolor =:black, legend = false)  
    end 
    mp4(anim, "anim_1D_$(gr).mp4", fps = 15)

end

# """
# 2D advection test
# """

solution2D  = Dict()

grid_reg  = RectilinearGrid(size = (N, N), x = Freg,  y = Freg,  halo = (3, 3), topology = (Periodic, Periodic, Flat), architecture = arch)    
grid_str  = RectilinearGrid(size = (N, N), x = Fsaw,  y = Fsaw,  halo = (3, 3), topology = (Periodic, Periodic, Flat), architecture = arch)    
grid_str2 = RectilinearGrid(size = (N, N), x = Fstr2, y = Fstr2, halo = (3, 3), topology = (Periodic, Periodic, Flat), architecture = arch)    

for (gr, grid) in enumerate([grid_reg, grid_str, grid_str2])
    
    U = Field{Face, Center, Center}(grid)
    V = Field{Center, Face, Center}(grid)

    parent(U) .= 1
    parent(V) .= 0.3
    
    Δt_max   = 0.1 * min_Δx(grid)  # faster in U than in V
    end_time = 2 * iter * Δt_max

    maxAdv1 = []; maxAdv2 = []; maxAdv3 = []; maxAdv4 = []; 
    for (adv, scheme) in enumerate(advection) 

        if adv == 2
            scheme = WENO5(grid = grid)
        end
        if adv == 3
            scheme = WENO5(grid = grid, stretched_smoothness = true)
        end
        if adv == 4
            scheme = WENO5(grid = grid, stretched_smoothness = true, zweno = true)
        end

        model = HydrostaticFreeSurfaceModel(        grid = grid,
                                                tracers  = :c,
                                        tracer_advection = scheme,
                                            velocities   = PrescribedVelocityFields(u=U, v=V), 
                                                coriolis = nothing,
                                                closure  = nothing,
                                                buoyancy = nothing)

        c = model.tracers.c
        set!(model, c=c₀_2D)
        
        simulation = Simulation(model,
                                Δt = Δt_max,
                                stop_time = end_time)

        for i = 1:end_time/Δt_max/10
            csim                               = adapt(CPU(), c)
            solution2D[(schemes[adv], Int(i))] = csim[1:N, 1:N, 1]
            
            for j = 1:10
                time_step!(model, Δt_max)
            end
        end
    end

    x     = adapt(CPU(), grid.xᶜᵃᵃ)[1:N]
    y     = adapt(CPU(), grid.yᵃᶜᵃ)[1:N]
    steps = ()
    anim = @animate for i ∈ 1:end_time/Δt_max/10
        plot(contourf(x, y, solution2D[(schemes[1], Int(i))], clim=(0, 1), levels = 0:0.1:1, title="Uweno"),
             contourf(x, y, solution2D[(schemes[2], Int(i))], clim=(0, 1), levels = 0:0.1:1, title="Sweno"),
             contourf(x, y, solution2D[(schemes[3], Int(i))], clim=(0, 1), levels = 0:0.1:1, title="βweno"),
             contourf(x, y, solution2D[(schemes[4], Int(i))], clim=(0, 1), levels = 0:0.1:1, title="Zweno"))
    end
    mp4(anim, "anim_2D_$gr.mp4", fps = 15)
end
