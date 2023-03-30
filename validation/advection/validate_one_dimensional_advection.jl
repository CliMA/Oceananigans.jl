using Oceananigans
using Oceananigans.Advection: AbstractCenteredAdvectionScheme, AbstractUpwindBiasedAdvectionScheme, VelocityStencil, VorticityStencil
using Oceananigans.Models.ShallowWaterModels: VectorInvariantFormulation, ConservativeFormulation
using JLD2
using OffsetArrays
using LinearAlgebra
using Plots
using Test
using Adapt 

@inline advection_order(buffer, ::Type{Centered})     = buffer * 2 
@inline advection_order(buffer, ::Type{UpwindBiased}) = buffer * 2 - 1  
@inline advection_order(buffer, ::Type{WENO})         = buffer * 2 - 1  

@inline form(a) = false
@inline form(::VectorInvariantFormulation) = VelocityStencil()

"""
This simulation is a simple 1D advection to test the 
validity of the stretched WENO scheme
"""

arch = CPU()

#parameters
N    = 100

# regular "stretched" grid
Freg = range(0, 1, length = N+1)

# center-coarsened grid
Δstr(i, N) = i < N/4 ? 1 : ( i > N*0.75 ? 1 : ( i < N/2 ? 1.08 * (i - N/4) + 1 : 1.08 * (3*N/4 - i) + 1  )  ) 
Fstr = zeros(Float64, N+1)
for i = 2:N+1
     Fstr[i] = Fstr[i-1] + Δstr(i-1, N)
end
Fstr ./= Fstr[end]

Freg = vcat(reverse(-Freg)[1:end-1], Freg)
Fstr = vcat(reverse(-Fstr)[1:end-1], Fstr)

Nx = 2N 
# solutions
solution  = Dict()
error     = Dict()

# 1D grid constructions
grid_reg  = RectilinearGrid(arch, size = (Nx, 1), x = Freg,  y = (0, 1), halo = (7, 7), topology = (Periodic, Periodic, Flat))    
grid_str  = RectilinearGrid(arch, size = (Nx, 1), x = Fstr,  y = (0, 1), halo = (7, 7), topology = (Periodic, Periodic, Flat))    

# the initial condition
@inline G(x, β, z) = exp(-β*(x - z)^2)
@inline F(x, α, a) = √(max(1 - α^2*(x-a)^2, 0.0))

Z = -0.7
δ = 0.005
β = log(2)/(36*δ^2)
a = 0.5
α = 10

@inline function c₀_1D(x, y, z) 
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

Schemes = [:Centered, :UpwindBiased, :WENO]

@inline grid_or_not(grid) = 1
@inline grid_or_not(::Nothing) = -1

# # Checking the accuracy of different schemes with different settings
buffers = [2, 3]
for (gr, grid) in enumerate([grid_str])
    
    @info "testing grid number $gr"

    Δt_max   = 0.2 * minimum_xspacing(grid)
    end_time = 2.0
    
    @show tot_iter = end_time ÷ Δt_max
    end_iter = tot_iter÷10
            
    c_real = CenterField(grid)
    formulation = ConservativeFormulation()
    
    for Scheme in [Schemes[2]]
        for buffer in buffers, gr in (nothing, grid)
                    
            scheme     = eval(Scheme)(gr, order = advection_order(buffer, eval(Scheme)))
            scheme_mom = eval(Scheme)(gr, order = advection_order(buffer, eval(Scheme))) #, vector_invariant = form(formulation))
            
            @info "Scheme $(summary(scheme_mom))" # with velocity $vel"
            model = ShallowWaterModel(; grid = grid,
                                     tracers = :c,
                          momentum_advection = scheme_mom,
                            tracer_advection = scheme,
                  gravitational_acceleration = 1.0,
                                    formulation)

            if formulation isa VectorInvariantFormulation
                set!(model, h=1.0, u=1.0, v=c₀_1D)
            else
                set!(model, h=1.0, uh=1.0, vh=c₀_1D)
            end

            set!(model, c=c₀_1D)
            c  = model.tracers.c
            vh = model.solution[2]
        
            for i = 1:end_iter
                for j = 1:10
                    time_step!(model, Δt_max)
                end
                csim  = Array(interior(vh, :, 1, 1))
                solution[(buffer, Int(i), grid_or_not(gr))] = csim
            end

            c_sol(x, y, z) = @. c₀_1D(x - model.clock.time + 2, y, z)
            set!(c_real, c_sol)
        end
        
        x = adapt(CPU(), grid.xᶜᵃᵃ)
        x = x[1:Nx]

        anim = @animate for i ∈ 1:end_iter
            plot(x, interior(c_real, :, 1, 1), seriestype=:scatter, ylims = (-0.3, 1.3), legend = false, title ="red: Centered, blue: Upwind, green: WENO") 
            plot!(x, solution[(buffers[1], Int(i), -1.0)], ylims = (-0.3, 1.3), linewidth = 1, linecolor =:red  , legend = false) 
            plot!(x, solution[(buffers[2], Int(i), -1.0)], ylims = (-0.3, 1.3), linewidth = 1, linecolor =:blue , legend = false) 
            plot!(x, solution[(buffers[3], Int(i), -1.0)], ylims = (-0.3, 1.3), linewidth = 1, linecolor =:green, legend = false)
            plot!(x, solution[(buffers[4], Int(i), -1.0)], ylims = (-0.3, 1.3), linewidth = 1, linecolor =:yellow, legend = false)
            plot!(x, solution[(buffers[1], Int(i), 1.0)], ylims = (-0.3, 1.3), linestyle=:dash, linewidth = 1, linecolor =:red  , legend = false) 
            plot!(x, solution[(buffers[2], Int(i), 1.0)], ylims = (-0.3, 1.3), linestyle=:dash, linewidth = 1, linecolor =:blue , legend = false) 
            plot!(x, solution[(buffers[3], Int(i), 1.0)], ylims = (-0.3, 1.3), linestyle=:dash, linewidth = 1, linecolor =:green, legend = false)
            plot!(x, solution[(buffers[4], Int(i), 1.0)], ylims = (-0.3, 1.3), linestyle=:dash, linewidth = 1, linecolor =:yellow, legend = false)
        end 
        mp4(anim, "anim_1D_$(gr)_$(Scheme).mp4", fps = 15)
    end
end
