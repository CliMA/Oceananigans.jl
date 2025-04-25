using Oceananigans, CairoMakie, Statistics
using Oceananigans.Models.NonhydrostaticModels: ConjugateGradientPoissonSolver
using Oceananigans.Solvers: DiagonallyDominantPreconditioner
using Oceananigans.Operators: ℑxyᶠᶜᵃ, ℑxyᶜᶠᵃ
using Oceananigans.Solvers: FFTBasedPoissonSolver
using Printf
using CUDA

using Oceananigans.BoundaryConditions: FlatExtrapolationOpenBoundaryCondition,
                                       PerturbationAdvectionOpenBoundaryCondition

# there is some problem using ConjugateGradientPoissonSolver with TimeInterval because the timestep can go really small
# so while I identify the issue I'm using IterationInterval and a fixed timestep

"""
    drag(modell; bounding_box = (-1, 3, -2, 2), ν = 1e-3)

Returns the drag within the `bounding_box` computed by:

∂ₜU + (U⋅∇)U = −∇P + ∇⋅τ + F⃗
F = ∫ᵥFdV = ∫ᵥ(∂U/∂t + (U⋅∇)U + ∇P − ∇⋅τ)dV
F = ∫ᵥ(∂ₜU)dV + ∮ₛ(U(U⋅n̂) + Pn̂ − τ⋅n̂)dS
Fᵤ = ∫ᵥ ∂ₜu dV + ∮ₛ(u(u⃗⋅n̂) − τₓₓ)dS + ∮ₛPx̂⋅dS⃗
Fᵤ = ∫ᵥ ∂ₜ u dV − ∫ₛ₁(u² − 2ν∂ₓ u + P)dS + ∫ₛ₂(u² − 2ν∂ₓ u+P)dS − ∫ₛ₃uvdS + ∫ₛ₄ uvdS

where the bounding box is ``V`` which is formed from the boundaries ``s1``, ``s2``, ``s3``, and ``s4``
which have outward directed normals ``-x̂``, ``x̂``, ``-ŷ``, and ``ŷ``

"""
function drag(model;
              bounding_box = (-1, 3, -2, 2),
              ν = 1e-3)

    u, v, _ = model.velocities

    uᶜ = Field(@at (Center, Center, Center) u)
    vᶜ = Field(@at (Center, Center, Center) v)

    xc, yc, _ = nodes(uᶜ)

    i₁ = findfirst(xc .> bounding_box[1])
    i₂ = findlast(xc .< bounding_box[2])

    j₁ = findfirst(yc .> bounding_box[3])
    j₂ = findlast(yc .< bounding_box[4])

    uₗ² = Field(uᶜ^2, indices = (i₁, j₁:j₂, 1))
    uᵣ² = Field(uᶜ^2, indices = (i₂, j₁:j₂, 1))

    uvₗ = Field(uᶜ*vᶜ, indices = (i₁:i₂, j₁, 1))
    uvᵣ = Field(uᶜ*vᶜ, indices = (i₁:i₂, j₂, 1))

    ∂₁uₗ = Field(∂x(uᶜ), indices = (i₁, j₁:j₂, 1))
    ∂₁uᵣ = Field(∂x(uᶜ), indices = (i₂, j₁:j₂, 1))

    ∂ₜuᶜ = Field(@at (Center, Center, Center) model.timestepper.Gⁿ.u)

    ∂ₜu = Field(∂ₜuᶜ, indices = (i₁:i₂, j₁:j₂, 1))

    p = model.pressures.pNHS

    ∫∂ₓp = Field(∂x(p), indices = (i₁:i₂, j₁:j₂, 1))

    a_local = Field(Integral(∂ₜu))

    a_flux = Field(Integral(uᵣ²)) - Field(Integral(uₗ²)) + Field(Integral(uvᵣ)) - Field(Integral(uvₗ))

    a_viscous_stress = 2ν * (Field(Integral(∂₁uᵣ)) - Field(Integral(∂₁uₗ)))

    a_pressure = Field(Integral(∫∂ₓp))

    return a_local + a_flux + a_pressure - a_viscous_stress
end

function cylinder_model(open_boundaries;

                        obc_name = "",

                        u∞ = 1,
                        r = 1/2,

                        cylinder = (x, y) -> ((x^2 + y^2) ≤ r^2),

                        stop_time = 100,

                        arch = GPU(),

                        Re = 100,
                        Ny = 512,
                        Nx = Ny,

                        ϵ = 0, # break up-down symmetry
                        x = (-6, 12), # 18
                        y = (-6 + ϵ, 6 + ϵ),  # 12

                        grid_kwargs = (; size=(Nx, Ny), x, y, halo=(6, 6), topology=(Bounded, Bounded, Flat)),

                        prefix = "flow_around_cylinder_Re$(Re)_Ny$(Ny)_$(obc_name)",

                        drag_averaging_window = 29)

    grid = RectilinearGrid(arch; grid_kwargs...)
    reduced_precision_grid = RectilinearGrid(arch, Float32; grid_kwargs...)

    grid = ImmersedBoundaryGrid(grid, GridFittedBoundary(cylinder))

    advection = Centered(order=2)
    closure = ScalarDiffusivity(ν=1/Re)

    no_slip = ValueBoundaryCondition(0)

    u_bcs = FieldBoundaryConditions(immersed=no_slip, east=open_boundaries.east, west=open_boundaries.west)

    v_bcs = FieldBoundaryConditions(immersed=no_slip,
                                    east=GradientBoundaryCondition(0),
                                    west=ValueBoundaryCondition(0))

    boundary_conditions = (u=u_bcs, v=v_bcs)

    preconditioner = FFTBasedPoissonSolver(reduced_precision_grid)
    reltol = abstol = 1e-7
    pressure_solver = ConjugateGradientPoissonSolver(grid, maxiter=10;
                                                    reltol, abstol, preconditioner)

    model = NonhydrostaticModel(; grid, pressure_solver, closure,
                                 advection, boundary_conditions)

    @show model

    uᵢ(x, y) = 1e-2 * randn()
    vᵢ(x, y) = 1e-2 * randn()
    set!(model, u=uᵢ, v=vᵢ)

    Δx = minimum_xspacing(grid)
    Δt = max_Δt = 0.2 * Δx^2 * Re

    simulation = Simulation(model; Δt, stop_time, minimum_relative_step = 1e-8)
    conjure_time_step_wizard!(simulation, cfl=0.7, IterationInterval(3); max_Δt)

    u, v, w = model.velocities

    # Drag computation
    drag_force = drag(model; ν=1/Re)
    compute!(drag_force)

    wall_time = Ref(time_ns())

    function progress(sim)
        if pressure_solver isa ConjugateGradientPoissonSolver
            pressure_iters = iteration(pressure_solver)
        else
            pressure_iters = 0
        end

        compute!(drag_force)
        D = CUDA.@allowscalar drag_force[1, 1, 1]
        cᴰ = D / (u∞ * r)
        vmax = maximum(model.velocities.v)

        msg = @sprintf("Iter: %d, time: %.2f, Δt: %.4f, Poisson iters: %d",
                    iteration(sim), time(sim), sim.Δt, pressure_iters)

        elapsed = 1e-9 * (time_ns() - wall_time[])

        msg *= @sprintf(", max v: %.2e, Cd: %0.2f, wall time: %s",
                        vmax, cᴰ, prettytime(elapsed))

        @info msg
        wall_time[] = time_ns()

        return nothing
    end

    add_callback!(simulation, progress, IterationInterval(100))

    ζ = ∂x(v) - ∂y(u)

    p = model.pressures.pNHS

    outputs = (; u, v, p, ζ)

    simulation.output_writers[:jld2] = JLD2Writer(model, outputs,
                                                  schedule = TimeInterval(0.1),
                                                  filename = prefix * "_fields.jld2",
                                                  overwrite_existing = true,
                                                  with_halos = true)

    simulation.output_writers[:drag] = JLD2Writer(model, (; drag_force),
                                                  schedule = TimeInterval(0.1),
                                                  filename = prefix * "_drag.jld2",
                                                  overwrite_existing = true,
                                                  with_halos = true,
                                                  indices = (1, 1, 1))

    run!(simulation)

    u = FieldTimeSeries(prefix * "_fields.jld2", "u")
    ζ = FieldTimeSeries(prefix * "_fields.jld2", "ζ")
    d = FieldTimeSeries(prefix * "_drag.jld2", "drag_force")

    Cd = ones(length(d)) .* NaN

    for n in drag_averaging_window+1:length(Cd)
        Cd[n] = - mean(d[1, 1, 1, n-29:n]) / r
    end

    n = Observable(1)

    title = @lift "Re 100, t = $(round(u.times[$n], digits = 2)), Cᴰ = $(round(Cd[$n], digits=2))"
    u_plt = @lift interior(u[$n], :, :, 1)
    ζ_plt = @lift interior(ζ[$n], :, :, 1)

    fig = Figure(size = (1500, 600))
    ax = Axis(fig[1, 1]; title = "x-velocity (m/s)", xlabel = "x (m)", ylabel = "y (m)", aspect = DataAspect())
    hm = heatmap!(ax, xnodes(u), ynodes(u), u_plt, colorrange = (0, 1.5), colormap = :batlowW)
    Colorbar(fig[1, 2], hm, label = "x-velocity (m/s)")

    ax2 = Axis(fig[1, 3]; title = "Vorticity (1/s)", xlabel = "x (m)", ylabel = "y (m)", aspect = DataAspect())
    hm2 = heatmap!(ax2, xnodes(ζ), ynodes(ζ), ζ_plt, colorrange = (-4, 4), colormap = :vik)
    Colorbar(fig[1, 4], hm2, label = "Vorticity (1/s)")

    supertitle = Label(fig[0, :], title)

    CairoMakie.record(fig, prefix"*.mp4", 1:length(u.times), framerate=20) do i
        @info "$i"
        n[] = i
    end

    return model, simulation
end

u∞ = 1

feobc = (east = FlatExtrapolationOpenBoundaryCondition(), west = OpenBoundaryCondition(u∞))

paobcs = (east = PerturbationAdvectionOpenBoundaryCondition(u∞; inflow_timescale = 1/4, outflow_timescale = Inf),
          west = PerturbationAdvectionOpenBoundaryCondition(u∞; inflow_timescale = 0.1, outflow_timescale = 0.1))

obcs = (; flat_extrapolation=feobc,
          perturbation_advection=paobcs)

for (obc_name, obc) in pairs(obcs)
    @info "Running $(obc_name)"
    cylinder_model(obc; obc_name, u∞)
end
