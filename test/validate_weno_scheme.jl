using Oceananigans
using Oceananigans.Grids: min_Δx, min_Δy, min_Δz
using JLD2
using OffsetArrays
using BenchmarkTools
using LinearAlgebra
using Plots
"""

This simulation is a simple 1D advection of a gaussian function, to test the 
validity of the stretched WENO scheme

"""
             
function multiple_steps!(model)
    for i = 1:20
        time_step!(model, 1e-6)
    end
    return nothing
end

N    = 32
arch = CPU()

# regular "stretched" mesh
Freg = range(0,1,length = N+1)


# seasaw mesh
Fsaw(j) = 1 / N  * (j - 1) + 0.1 * 1 / N * mod(j - 1, 2)
  
function Δstr2(i, N)
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

 Fstr2 = zeros(Float64, N+1)

 for i = 2:N+1
     Fstr2[i] = Fstr2[i-1] + Δstr2(i-1, N)
 end

 Fstr2 ./= Fstr2[end]

solution  = Dict()
real_sol  = Dict()
coord     = Dict()
residual  = Dict()
time      = Dict()

grid_reg  = RectilinearGrid(size = (N,), x = Freg,  halo = (3,), topology = (Periodic, Flat, Flat), architecture = arch)    
grid_str  = RectilinearGrid(size = (N,), x = Fsaw,  halo = (3,), topology = (Periodic, Flat, Flat), architecture = arch)    
grid_str2 = RectilinearGrid(size = (N,), x = Fstr2, halo = (3,), topology = (Periodic, Flat, Flat), architecture = arch)    

advection = [WENO5(), WENO5(), WENO5()]
# advection = [WENO5()]

schemes = [:wreg, :wstr, :wstrS]

vel = 1
# Checking the accuracy of different schemes with different settings

for grid in [grid_reg, grid_str, grid_str2]
    grid == grid_reg ? gr = :reg : grid == grid_str ? gr = :str : gr = :str2

    U = Field(Face, Center, Center, arch, grid)
    parent(U) .= vel

    x        = grid.xᶜᵃᵃ[1:grid.Nx]

    coord[(gr)] = x
    Δt_max      = 0.2 * min_Δx(grid)
    end_time    = 1000 * Δt_max
    c₀(x, y, z) = 10*exp(-((x-0.5)/0.1)^2)
                                            
    for (adv, scheme) in enumerate(advection) 
        if adv == 2
            scheme = WENO5(grid = grid)
        end
        if adv == 3
            scheme = WENO5(grid = grid, stretched_smoothness = true)
        end

        model = HydrostaticFreeSurfaceModel(architecture = arch,
                                                    grid = grid,
                                                tracers = :c,
                                        tracer_advection = scheme,
                                            velocities = PrescribedVelocityFields(u=U), 
                                                coriolis = nothing,
                                                closure = nothing,
                                                buoyancy = nothing)

        
        set!(model, c=c₀)
        c = model.tracers.c

        simulation = Simulation(model,
                                Δt = Δt_max,
                                stop_time = end_time)                           
        
        
        for i = 1:end_time/Δt_max
            ctest   = Array(parent(model.tracers.c.data))
            offsets = (model.tracers.c.data.offsets[1],  model.tracers.c.data.offsets[2],  model.tracers.c.data.offsets[3])
            ctemp   = OffsetArray(ctest, offsets)
            ctests  = ctemp[1:grid.Nx, 1 ,1]
            creal    = Array(c₀.(mod.((x .- vel * (i-1) * 10 * Δt_max),1), 0, 0))
            solution[(schemes[adv], gr, Int(i))] = ctests
            real_sol[(gr, Int(i))] = creal
            for j = 1:10
                time_step!(model, Δt_max)
            end
        end
        
        time[(schemes[adv], gr)] = @belapsed multiple_steps!($model)

    end
    
    anim = @animate for i ∈ 1:end_time/Δt_max
        plot(x, real_sol[(gr, Int(i))], seriestype=:scatter, legend = false, title ="red: Uweno, blue: Sweno, green: βweno") 
        plot!(x, solution[(schemes[1], gr, Int(i))], linewidth = 1, linecolor =:red  , legend = false) 
        plot!(x, solution[(schemes[2], gr, Int(i))], linewidth = 1, linecolor =:blue , legend = false) 
        plot!(x, solution[(schemes[3], gr, Int(i))], linewidth = 1, linecolor =:green, legend = false) 
    end 
    gif(anim, "anim_1D_$(gr)_right.mp4", fps = 15)
    
end

# pos   = Dict()
# plots = Dict()
# for grid in [grid_reg, grid_str, grid_str2]
#     grid == grid_reg ? gr = :reg : grid == grid_str ? gr = :str : gr = :str2
#     plots[(gr)] = plot(coord[(gr)], real_sol[(gr, Int(end_time/Δt_max/10))],seriestype=:scatter )
#     for adv = 1:3
#         plots[(gr)] = plot!(coord[(gr)], solution[(schemes[adv], gr, Int(end_time/Δt_max/10))])
#     end
# end

# """
# Now test 2D advection 
# # """

solution2D  = Dict()
real_sol2D  = Dict()
coord2D     = Dict()
residual2D  = Dict()

grid_reg  = RectilinearGrid(size = (N, N), x = Freg,  z = Freg,  halo = (3, 3), topology = (Periodic, Flat, Periodic), architecture = arch)    
grid_str  = RectilinearGrid(size = (N, N), x = Fsaw,  z = Fsaw,  halo = (3, 3), topology = (Periodic, Flat, Periodic), architecture = arch)    
grid_str2 = RectilinearGrid(size = (N, N), x = Fstr2, z = Fstr2, halo = (3, 3), topology = (Periodic, Flat, Periodic), architecture = arch)    

for grid in [grid_reg, grid_str, grid_str2]
    
    grid == grid_reg ? gr = :reg : grid == grid_str ? gr = :str : gr = :str2

    U = Field(Face, Center, Center, arch, grid)
    V = Field(Center, Center, Face, arch, grid)

    parent(U) .= vel
    parent(V) .= 0.3
    
    Δt_max   = 0.1 * min_Δx(grid) / 2^(0.5)
    end_time = 3000 * Δt_max

    for (adv, scheme) in enumerate(advection) 

        if adv == 2
            scheme = WENO5(grid = grid)
        end

        if adv == 3
            scheme = WENO5(grid = grid, stretched_smoothness = true)
        end

        model = HydrostaticFreeSurfaceModel(architecture = arch,
                                                    grid = grid,
                                                tracers  = :c,
                                        tracer_advection = scheme,
                                            velocities   = PrescribedVelocityFields(u=U, w=V), 
                                                coriolis = nothing,
                                                closure  = nothing,
                                                buoyancy = nothing)


        mask(y) = (y < 0.6 && y > 0.4) ? 1 : 0
        # c₀(x, y, z) = 10*exp(-((x-0.5)/0.3)^2) * mask(y)
        # c₀(x, y, z) = 10*exp(-((y-0.5)/0.3)^2) * mask(z)
        c₀(x, y, z) = 1 * mask(z) * mask(x) 
                                
        c = model.tracers.c
        parent(c) .= 0
        set!(model, c=c₀)
        
        simulation = Simulation(model,
                                Δt = Δt_max,
                                stop_time = end_time)                           

        for i = 1:end_time/Δt_max/10
            ctest   = Array(parent(model.tracers.c.data))
            offsets = (model.tracers.c.data.offsets[1],  model.tracers.c.data.offsets[2],  model.tracers.c.data.offsets[3])
            ctemp   = OffsetArray(ctest, offsets)
            # ctests  = ctemp[1 ,1:grid.Ny, 1:grid.Nz]
            ctests  = ctemp[1:grid.Nx, 1 ,1:grid.Nz]
            # ctests  = ctemp[1:grid.Nx, 1:grid.Ny, 1]
            solution2D[(schemes[adv], gr, Int(i))] = ctests
            for j = 1:10
                time_step!(model, Δt_max)
            end
        end

    end

    x        = grid.xᶜᵃᵃ[1:grid.Nx]
    # y        = grid.yᵃᶜᵃ[1:grid.Ny]
    y        = grid.zᵃᵃᶜ[1:grid.Nz]
    coord2D[(gr)]    = (x, y)
    anim = @animate for i ∈ 1:end_time/Δt_max/10
        plot(contourf(x, y, solution2D[(schemes[1], gr, Int(i))], clim=(0, 1), levels = 0:0.1:1),
             contourf(x, y, solution2D[(schemes[2], gr, Int(i))], clim=(0, 1), levels = 0:0.1:1),
             contourf(x, y, solution2D[(schemes[3], gr, Int(i))], clim=(0, 1), levels = 0:0.1:1))
    end 
    gif(anim, "anim_2D_$gr.mp4", fps = 15)
end

