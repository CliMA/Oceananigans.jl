
using Oceananigans.Grids: min_Δx
using Oceananigans.Advection: WENO5S
using JLD2
using Plots

""" cosine grid """

Nx = 40
Δx = 0.02 .+ 2*sin.(π*collect(1:Nx)/(Nx))

xF = zeros(Float64, Nx+1)

xF[1] = 0
for i in 2:Nx+1
    xF[i] = xF[i-1] + Δx[i-1]
end

xF = xF ./ xF[end]

grid  = RectilinearGrid(size = (Nx,), x = xF, halo = (3,), topology = (Periodic, Flat, Flat), architecture = GPU())    


U = Field(Face, Center, Center, grid)
set!(U, 1.0)


for weno in [WENO5(), WENO5S()]
    model = HydrostaticFreeSurfaceModel(architecture = GPU(),
                                                grid = grid,
                                           advection = weno,
                                         timestepper = :RungeKutta3,
                                             tracers = (:c,),
                                          velocities = PrescribedVelocityFields(u=U,),
                                            buoyancy = nothing)


    c₀(x, y, z) = 10*exp(-((x-0.5)/0.2)^2)

    set!(model, c=c₀, u=0, v=0, w=0)

    c = model.tracers.c

    function progress(s)
        c = s.model.tracers.c
        @info "Maximum(c) = $(maximum(c)), time = $(s.model.clock.time) / $(s.stop_time)"
        return nothing
    end

    Δ_min = min_Δx(grid)

    # Time-scale for advection propagation across the smallest grid cell
    CFL    = 0.5
    Δt_max = CFL * Δ_min

    simulation = Simulation(model,
                            Δt = Δt_max,
                            stop_time = 4000*Δt_max)

                                                            
    output_fields = model.tracers

    output_prefix = "test_weno_$weno"

    simulation.output_writers[:fields] = JLD2OutputWriter(model, output_fields,
                                                        schedule = TimeInterval(5*Δt_max),
                                                        prefix = output_prefix,
                                                        force = true)

    simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

    run!(simulation, pickup=false)
end



wenoU = jldopen("test_weno_WENO5().jld2")
wenoS = jldopen("test_weno_WENO5S().jld2")

global cu = ()
global cs = ()

for (i, key) in enumerate(keys(wenoU["timeseries/c"]))
    if i > 1
        global cu = (cu..., wenoU["timeseries/c/$key"])
        global cs = (cs..., wenoS["timeseries/c/$key"])
    end
end

x = grid.xᶜᵃᵃ[1:grid.Nx]

anim = @animate for i ∈ 1:length(cu)
    plot( x, cu[i][:])
    plot!(x, cs[i][:])
end
gif(anim, "anim_fps15.gif", fps = 15)
