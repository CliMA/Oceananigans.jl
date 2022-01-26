using Printf
using CUDA
using Oceananigans
using Oceananigans.TurbulenceClosures: ExplicitTimeDiscretization, VerticallyImplicitTimeDiscretization
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary
using LinearAlgebra

function boundary_clustered(N, L, ini)
    Δz(k)   = k < N / 2 + 1 ? 2 / (N - 1) * (k - 1) + 1 : - 2 / (N - 1) * (k - N) + 1 
    z_faces = zeros(N+1) 
    for k = 2:N+1
        z_faces[k] = z_faces[k-1] + Δz(k-1)
    end
    z_faces = z_faces ./ z_faces[end] .* L .+ ini
    return z_faces
end

function center_clustered(N, L, ini)
    Δz(k)   = k < N / 2 + 1 ? 2 / (N - 1) * (k - 1) + 1 : - 2 / (N - 1) * (k - N) + 1 
    z_faces = zeros(N+1) 
    for k = 2:N+1
        z_faces[k] = z_faces[k-1] + 3 - Δz(k-1)
    end
    z_faces = z_faces ./ z_faces[end] .* L .+ ini
    return z_faces
end

grid = RectilinearGrid(GPU(), size=(512, 256), 
                       x = (-10, 10), 
                       z = (0, 5),
                topology = (Periodic, Flat, Bounded))

# Gaussian bump of width "1"
bump(x, y, z) = z < exp(-x^2)

@inline show_name(t) = t isa ExplicitTimeDiscretization ? "explicit" : "implicit"

grid_with_bump = ImmersedBoundaryGrid(grid, GridFittedBoundary(bump))

# Tidal forcing
tidal_forcing(x, y, z, t) = 1e-4 * cos(t)

for time_stepper in (ExplicitTimeDiscretization(), VerticallyImplicitTimeDiscretization())
    
    model = HydrostaticFreeSurfaceModel(grid = grid_with_bump,
                                        momentum_advection = CenteredSecondOrder(),
                                        free_surface = ExplicitFreeSurface(gravitational_acceleration=10),
                                        closure = IsotropicDiffusivity(ν=1e-2, κ=1e-2, time_discretization = time_stepper),
                                        tracers = :b,
                                        buoyancy = BuoyancyTracer(),
                                        coriolis = FPlane(f=sqrt(0.5)),
                                        forcing = (u = tidal_forcing,))

    # Linear stratification
    set!(model, b = (x, y, z) -> 4 * z)

    progress_message(s) = @info @sprintf("[%.2f%%], iteration: %d, time: %.3f, max|w|: %.2e",
                                100 * s.model.clock.time / s.stop_time, s.model.clock.iteration,
                                s.model.clock.time, maximum(abs, model.velocities.w))

    gravity_wave_speed = sqrt(model.free_surface.gravitational_acceleration * grid.Lz)
    
    Δt = CUDA.@allowscalar 0.1 * minimum(grid.Δxᶜᵃᵃ) / gravity_wave_speed
    
    simulation = Simulation(model, Δt = Δt, stop_time = 50000Δt)

    serialize_grid(file, model) = file["serialized/grid"] = model.grid.grid

    simulation.output_writers[:fields] = JLD2OutputWriter(model, merge(model.velocities, model.tracers),
                                                        schedule = TimeInterval(0.1),
                                                        prefix = "internal_tide_$(show_name(time_stepper))",
                                                        init = serialize_grid,
                                                        force = true)

    simulation.callbacks[:progress] = Callback(progress_message, IterationInterval(10))

    run!(simulation)

    @info """
        Simulation complete.
        Output: $(abspath(simulation.output_writers[:fields].filepath))
    """
end

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

    grid = adapt(CPU(), file["serialized/grid"])

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

    mp4(anim, prefix * ".mp4", fps = 16)

    close(file)
end

function plot_implicit_explicit_difference(filename)

    file_explicit = jldopen(filename * "_explicit.jld2")
    file_implicit = jldopen(filename * "_implicit.jld2")

    iterations = parse.(Int, keys(file_explicit["timeseries/t"]))   

    comparison_u = zeros(length(iterations))
    comparison_w = zeros(length(iterations))
    comparison_b = zeros(length(iterations))

    for (i, iter) in enumerate(iterations)

        u_explicit = file_explicit["timeseries/u/$iter"][:, 1, :]
        w_explicit = file_explicit["timeseries/w/$iter"][:, 1, :]
        b_explicit = file_explicit["timeseries/b/$iter"][:, 1, :]

        u_implicit = file_implicit["timeseries/u/$iter"][:, 1, :]
        w_implicit = file_implicit["timeseries/w/$iter"][:, 1, :]
        b_implicit = file_implicit["timeseries/b/$iter"][:, 1, :]
        
        comparison_u[i] = norm(u_explicit .- u_implicit) 
        comparison_w[i] = norm(w_explicit .- w_implicit) 
        comparison_b[i] = norm(b_explicit .- b_implicit) 
    end

    kwargs = (linewidth = 2, foreground_color_legend = nothing, legendfontsize = 12, legend = :right, grid = false,
              xtickfontsize = 12, ytickfontsize=12, xlabel = "time", ylabel = "norm of difference")

     plot(iterations, comparison_u, label = "u"; kwargs...)
    plot!(iterations, comparison_w, label = "w"; kwargs...)
    plot!(iterations, comparison_b, label = "b"; kwargs...)

    savefig(filename * "_comparison_implicit_explicit.png")

    close(file_explicit)
    close(file_implicit)
end

visualize_internal_tide_simulation("internal_tide_explicit")
visualize_internal_tide_simulation("internal_tide_implicit")
plot_implicit_explicit_difference("internal_tide")
