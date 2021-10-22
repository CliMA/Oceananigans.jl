using Printf
using Oceananigans
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary

grid = RectilinearGrid(size=(512, 256), x=(-10, 10), z=(0, 5), topology=(Periodic, Flat, Bounded))

# Gaussian bump of width "1"
bump(x, y, z) = z < exp(-x^2)

grid_with_bump = ImmersedBoundaryGrid(grid, GridFittedBoundary(bump))

# Tidal forcing
tidal_forcing(x, y, z, t) = 1e-4 * cos(t)

model = HydrostaticFreeSurfaceModel(architecture = GPU(),
                                    grid = grid_with_bump,
                                    momentum_advection = CenteredSecondOrder(),
                                    free_surface = ExplicitFreeSurface(gravitational_acceleration=10),
                                    closure = IsotropicDiffusivity(ν=1e-4, κ=1e-4),
                                    tracers = :b,
                                    buoyancy = BuoyancyTracer(),
                                    coriolis = FPlane(f=sqrt(0.5)),
                                    forcing = (u = tidal_forcing,))

# Linear stratification
set!(model, b = (x, y, z) -> 4 * z)

progress(s) = @info @sprintf("[%.2f%%], iteration: %d, time: %.3f, max|w|: %.2e",
                             100 * s.model.clock.time / s.stop_time, s.model.clock.iteration,
                             s.model.clock.time, maximum(abs, model.velocities.w))

gravity_wave_speed = sqrt(model.free_surface.gravitational_acceleration * grid.Lz)
Δt = 0.1 * grid.Δx / gravity_wave_speed
              
simulation = Simulation(model, Δt = Δt, stop_time = 100, progress = progress, iteration_interval = 100)

serialize_grid(file, model) = file["serialized/grid"] = model.grid.grid

simulation.output_writers[:fields] = JLD2OutputWriter(model, merge(model.velocities, model.tracers),
                                                      schedule = TimeInterval(0.1),
                                                      prefix = "internal_tide",
                                                      init = serialize_grid,
                                                      force = true)
                        
run!(simulation)

@info """
    Simulation complete.
    Output: $(abspath(simulation.output_writers[:fields].filepath))
"""

using JLD2
using Plots
ENV["GKSwstype"] = "100"

function nice_divergent_levels(c, clim; nlevels=20)
    levels = range(-clim, stop=clim, length=nlevels)
    cmax = maximum(abs, c)
    clim < cmax && (levels = vcat([-cmax], levels, [cmax]))
    return (-clim, clim), levels
end

function nan_solid(x, z, u, bump)
    Nx, Nz = size(u)
    x2 = reshape(x, Nx, 1)
    z2 = reshape(z, 1, Nz)
    u[bump.(x2, 0, z2)] .= NaN
    return nothing
end

function visualize_internal_tide_simulation(prefix)

    filename = prefix * ".jld2"
    file = jldopen(filename)

    grid = file["serialized/grid"]

    bump(x, y, z) = z < exp(-x^2)

    xu, yu, zu = nodes((Face, Center, Center), grid)
    xw, yw, zw = nodes((Center, Center, Face), grid)
    xb, yb, zb = nodes((Center, Center, Center), grid)

    b₀ = file["timeseries/b/0"][:, 1, :]

    iterations = parse.(Int, keys(file["timeseries/t"]))    

    anim = @animate for (i, iter) in enumerate(iterations)

        @info "Plotting iteration $iter of $(iterations[end])..."

        u = file["timeseries/u/$iter"][:, 1, :]
        w = file["timeseries/w/$iter"][:, 1, :]
        b = file["timeseries/b/$iter"][:, 1, :]
        t = file["timeseries/t/$iter"]

        b′ = b .- b₀

        wlims, wlevels = nice_divergent_levels(w, 1e-4)
        ulims, ulevels = nice_divergent_levels(u, 1e-3)
        blims, blevels = nice_divergent_levels(b′, 1e-4)
        
        nan_solid(xu, zu, u, bump)
        nan_solid(xw, zw, w, bump)
        nan_solid(xb, zb, b, bump) 

        u_title = @sprintf("x velocity, t = %.2f", t)

        u_plot = contourf(xu, zu, u'; title = u_title,                  color = :balance, aspectratio = :equal, linewidth = 0, levels = ulevels, clims = ulims)
        w_plot = contourf(xw, zw, w'; title = "z velocity",             color = :balance, aspectratio = :equal, linewidth = 0, levels = wlevels, clims = wlims)
        b_plot = contourf(xb, zb, b′'; title = "buoyancy perturbation", color = :balance, aspectratio = :equal, linewidth = 0, levels = blevels, clims = blims)

        plot(u_plot, w_plot, b_plot, layout = (3, 1), size = (1200, 1200))
    end

    mp4(anim, "internal_tide.mp4", fps = 16)

    close(file)
end

visualize_internal_tide_simulation("internal_tide")
