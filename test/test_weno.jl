
using Oceananigans.Grids: min_Δz
using Oceananigans.Advection: WENO5S
using JLD2
using Plots

grid  = RectilinearGrid(size = (100,), z = (0, 1), halo = (3,), topology = (Flat, Flat, Periodic))    

wback(x, y, z, t) = 1.0

W = BackgroundField(wback)


for weno in [WENO5S()]
    model = NonhydrostaticModel(architecture = CPU(),
                                        grid = grid,
                                advection = weno,
                                timestepper = :RungeKutta3,
                                    tracers = (:c,),
                        background_fields = (w = W, ),
                                    buoyancy = nothing)


    c₀(x, y, z) = 10*exp(-((z-0.5)/0.2)^2)

    set!(model, c=c₀, u=0, v=0, w=0)

    c = model.tracers.c

    function progress(s)
        c = s.model.tracers.c
        @info "Maximum(c) = $(maximum(c)), time = $(s.model.clock.time) / $(s.stop_time)"
        return nothing
    end

    Δ_min = min_Δz(grid)

    # Time-scale for advection propagation across the smallest grid cell
    CFL    = 0.5
    Δt_max = CFL * Δ_min

    simulation = Simulation(model,
                            Δt = Δt_max,
                            stop_time = 20000*Δt_max)

                                                            
    output_fields = model.tracers

    output_prefix = "test_weno_fields$(grid.Nx)_$weno"

    simulation.output_writers[:fields] = JLD2OutputWriter(model, output_fields,
                                                        schedule = TimeInterval(100*Δt_max),
                                                        prefix = output_prefix,
                                                        force = true)

    simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

    run!(simulation, pickup=false)
end



wenoU = jldopen("test_weno_fields1_WENO5().jld2")
wenoS = jldopen("test_weno_fields1_WENO5S().jld2")

global cu = ()
global cs = ()

for (i, key) in enumerate(keys(wenoU["timeseries/c"]))
    if i > 1
        global cu = (cu..., wenoU["timeseries/c/$key"])
        global cs = (cs..., wenoS["timeseries/c/$key"])
    end
end

z = grid.zᵃᵃᶜ[1:grid.Nz]

# anim = @animate for i ∈ 1:length(cu)
#     plot(z, cu[i][1,1,:], i)
# end
# gif(anim, "anim_fps15.gif", fps = 15)
