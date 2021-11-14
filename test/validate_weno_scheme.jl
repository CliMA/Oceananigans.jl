using Oceananigans
using Oceananigans.Grids: min_Δx
using Oceananigans.Advection: WENO5S
using JLD2
using Plots
using BenchmarkTools
using Test
using LinearAlgebra

Nx = 20

architectures = [GPU(), CPU()]

# background advection velocity
wback(x, y, z, t) = 1.0
U = BackgroundField(wback)

# regular "stretched" mesh
xF_reg = range(0,1,length = Nx+1)

#stretched mesh
Δx = zeros(Nx); Δx .= 100
Δx[4] = 65; Δx[5] = 12; Δx[6] = 48; Δx[7] = 208; Δx[8] = 15

xF_str = zeros(Float64, Nx+1)
xF_str[1] = 0
for i in 2:Nx+1
    xF_str[i] = xF_str[i-1] + Δx[i-1]
end
xF_str = xF_str ./ xF_str[end]

#chebishev grid
@inline xF_che(i) = 1/2 + 1/2 * cos(π * (i - 1) / Nx);

# Test accuracy
for arch in architectures, xF in [xF_reg, xF_str, xF_che]

    if xF == xF_che
        grid = RectilinearGrid(size = (Nx,), x = xF, halo = (3,), topology = (Bounded, Flat, Flat), architecture = arch)    
    else
        grid = RectilinearGrid(size = (Nx,), x = xF, halo = (3,), topology = (Periodic, Flat, Flat), architecture = arch)    
    end

    x = grid.xᶜᵃᵃ[1:grid.Nx]

    Δt_max = 0.75 * min_Δx(grid)

    c₀(x, y, z) = 10*exp(-((x-0.5)/0.2)^2)
    advection    = [WENO5S(grid = grid), WENO5S(), WENO5()]

    residual = zeros(Float64, 3)

    for (adv, weno) in enumerate(advection)
        model = NonhydrostaticModel(architecture = arch,
                                        grid = grid,
                                   advection = weno,
                                     tracers = (:c,),
                                 timestepper = :RungeKutta3,
                           background_fields = (u=U,),
                                    buoyancy = nothing)

        set!(model, c=c₀, u=0, v=0, w=0)
        c = model.tracers.c

        end_time   = 4000 * Δt_max

        simulation = Simulation(model,
                                Δt = Δt_max,
                                stop_time = end_time)                           
        run!(simulation, pickup=false)

        residal[adv] = norm(c₀.(mod.((x .- end_time),1), 0, 0) - c[1:grid.Nx, 1, 1])
    end

    @info """
        residuals for settings
            architecture is $(typeof(arch ))
            the spacing is $(xF == xF_reg ? "regular" : "stretched")

            WENO5S stretched setting : $(residual[1]), 
            WENO5S   uniform setting : $(residual[2]), 
            WENO5     (only uniform) : $(residual[3])
        """
end


function multiple_steps!(model)
    for i = 1:20
        time_step!(model, 1e-6)
    end
    return nothing
end

# Benchmark time required
for arch in architectures, xF in [xF_reg, xF_str, xF_che]

    if xF == xF_che
        grid = RectilinearGrid(size = (Nx,), x = xF, halo = (3,), topology = (Bounded, Flat, Flat), architecture = arch)    
    else
        grid = RectilinearGrid(size = (Nx,), x = xF, halo = (3,), topology = (Periodic, Flat, Flat), architecture = arch)    
    end

    x = grid.xᶜᵃᵃ[1:grid.Nx]

    Δt_max = 0.75 * min_Δx(grid)

    c₀(x, y, z) = 10*exp(-((x-0.5)/0.2)^2)
    advection    = [WENO5S(grid = grid), WENO5S(), WENO5()]

    time = ()
    for weno in advection

        model = NonhydrostaticModel(architecture = arch,
                                            grid = grid,
                                    advection = weno,
                                        tracers = (:c,),
                            background_fields = (u=U,),
                                        buoyancy = nothing)


        c₀(x, y, z) = 10*exp(-((x-0.5)/0.2)^2)

        set!(model, c=c₀)

        for i=1:10
            time_step!(model, 1e-6) # warmup
        end
        time = (time..., @btime multiple_steps!($model))
    end

    @info """
        benchmark for settings
            architecture is $(typeof(arch ))
            the spacing is $(xF == xF_reg ? "regular" : "stretched")

            WENO5S stretched setting : $(time[1]), 
            WENO5S   uniform setting : $(time[2]), 
            WENO5     (only uniform) : $(time[3])
        """
end
