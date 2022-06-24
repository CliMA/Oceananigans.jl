using Oceananigans
using Oceananigans.Advection: AbstractCenteredAdvectionScheme, AbstractUpwindBiasedAdvectionScheme
using Oceananigans.Grids: min_Δx, min_Δy, min_Δz
using JLD2
using OffsetArrays
using LinearAlgebra
using Plots
using Test
using Adapt 

@inline advection_order(buffer, ::Type{Centered})     = buffer * 2 
@inline advection_order(buffer, ::Type{UpwindBiased}) = buffer * 2 - 1  
@inline advection_order(buffer, ::Type{WENO})         = buffer * 2 - 1  

"""
This simulation is a simple 1D advection to test the 
validity of the stretched WENO scheme
"""

#parameters
N    = 100
arch = CPU()

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
error     = Dict()

# 1D grid constructions
grid_reg  = RectilinearGrid(arch, size = N, x = Freg,  halo = 7, topology = (Periodic, Flat, Flat))    
grid_str  = RectilinearGrid(arch, size = N, x = Fsaw,  halo = 7, topology = (Periodic, Flat, Flat))    
grid_str2 = RectilinearGrid(arch, size = N, x = Fstr2, halo = 7, topology = (Periodic, Flat, Flat))    

# mask for the initial condition
mask(x) = Int(x > 0.35 && x < 0.65)
c₀_1D(x, y, z) = @. 10 * mask(x) #exp(-(x - 0.5)^2 / 0.2^2)

Schemes = [:Centered, :UpwindBiased, :WENO]

@inline grid_or_not(grid) = 1
@inline grid_or_not(::Nothing) = 2

# Checking the accuracy of different schemes with different settings
for (gr, grid) in enumerate([grid_reg, grid_str, grid_str2])
    
    @info "testing grid number $gr"

    Δt_max   = 0.2 * min_Δx(grid)
    end_time = 2.0
    
    @show tot_iter = end_time ÷ Δt_max
    end_iter = tot_iter÷10
            
    c_real = CenterField(grid)
    
    for buffer in [1, 2, 3, 4, 5, 6]
        for Scheme in Schemes[1:3], advection_grid in [nothing, grid]
            
            U = Field{Face, Center, Center}(grid)
            parent(U) .= 1
        
            scheme = eval(Scheme)(advection_grid, order = advection_order(buffer, eval(Scheme)))
            @info "Scheme $(summary(scheme))" # with velocity $vel"

            model  = HydrostaticFreeSurfaceModel(       grid = grid,
                                                    tracers = :c,
                                            tracer_advection = scheme,
                                                velocities = PrescribedVelocityFields(u=U), 
                                                    coriolis = nothing,
                                                    closure = nothing,
                                                    buoyancy = nothing)
            
            set!(model, c=c₀_1D)
            c = model.tracers.c
        
            for i = 1:end_iter
                csim  = Array(interior(c, :, 1, 1))
                solution[(Scheme, Int(i), grid_or_not(advection_grid))] = csim
                for j = 1:10
                    time_step!(model, Δt_max)
                end
            end

            c_sol(x, y, z) = @. c₀_1D(x - model.clock.time, y, z)
            set!(c_real, c_sol)

            error[(Scheme, buffer, grid_or_not(advection_grid), gr)] = norm(solution[(Scheme, Int(end_iter), grid_or_not(advection_grid))] .- Array(interior(c_real, :, 1, 1)), 2) / grid.Nx^(1/2)
        end
        
        x = adapt(CPU(), grid.xᶜᵃᵃ)
        x = x[1:N]

        anim = @animate for i ∈ 1:end_time
            plot(x, c₀_1D(x .- Δt_max * i * 10, 1, 1), seriestype=:scatter, ylims = (-3, 13), legend = false, title ="red: Centered, blue: Upwind, green: WENO, buffer = $buffer") 
            plot!(x, solution[(Schemes[1], Int(i), 1)], ylims = (-3, 13), linewidth = 1, linecolor =:red  , legend = false) 
            plot!(x, solution[(Schemes[2], Int(i), 1)], ylims = (-3, 13), linewidth = 1, linecolor =:blue , legend = false) 
            plot!(x, solution[(Schemes[3], Int(i), 1)], ylims = (-3, 13), linewidth = 1, linecolor =:green, legend = false)
            plot!(x, solution[(Schemes[1], Int(i), 2)], ylims = (-3, 13), linestyle=:dash, linewidth = 1, linecolor =:red  , legend = false) 
            plot!(x, solution[(Schemes[2], Int(i), 2)], ylims = (-3, 13), linestyle=:dash, linewidth = 1, linecolor =:blue , legend = false) 
            plot!(x, solution[(Schemes[3], Int(i), 2)], ylims = (-3, 13), linestyle=:dash, linewidth = 1, linecolor =:green, legend = false)
        end 
        mp4(anim, "anim_1D_$(gr)_$(buffer).mp4", fps = 15)
    end
end
