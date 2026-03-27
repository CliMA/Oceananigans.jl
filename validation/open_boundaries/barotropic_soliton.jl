"""
Hydrostatic equatorial Rossby soliton simulation following Haidvogel and
Beckmann (1999), section 6.1, used by Marchesiello, McWilliams, & Shchepetkin (2001)
to test their open boundary conditions.

All model variables are dimensional (SI units: m, s, m/s).
Non-dimensionalization follows H = 40 cm, L = 295 km, U = sqrt(g*H) ≈ 1.981 m/s,
T = L/U.  The non-dimensional soliton equations are evaluated internally and the
results are multiplied by the appropriate dimensional scales before passing to
Oceananigans.  With this scaling, the default gravitational_acceleration = 9.81 m/s²
is correct and no override is needed.

The analytical solution uses perturbation expansions from Boyd (1980, 1985).
Runs the simulation with a β-plane Coriolis force, saves JLD2 output, then
animates the numerical solution side by side with the analytical solution.

References:
    Boyd, J. P. (1980). Equatorial Solitary Waves. Part 1: Rossby Solitons.
        Journal of Physical Oceanography, 10, 1699–1717.
    Boyd, J. P. (1985). Equatorial Solitary Waves. Part 3: Westward-Traveling
        Modons. Journal of Physical Oceanography, 15, 46–54.
    Haidvogel, D. B., and A. Beckmann (1999). Numerical Ocean Circulation
        Modeling. Imperial College Press, section 6.1.
"""

using Oceananigans
using Oceananigans.Units
using CairoMakie
using Printf

g = Oceananigans.defaults.gravitational_acceleration

#+++ Soliton analytical solution
# Haidvogel & Beckmann (1999), section 6.1; following Boyd (1980, 1985)
struct SolitonParameters
    # Physical scales (SI units)
    H::Float64  # Depth scale [m]
    L::Float64  # Length scale [m]
    T::Float64  # Time scale [s]
    U::Float64  # Velocity scale [m/s] = sqrt(g * H)

    # Dimensional domain bounds [m]
    x_min::Float64
    x_max::Float64
    y_min::Float64
    y_max::Float64

    # Non-dimensional soliton parameters (unitless)
    B::Float64      # Soliton amplitude parameter (should be < 0.6)
    A::Float64      # Amplitude coefficient A = 0.771 * B²
    c⁰::Float64     # Zero-order wave speed c^(0) = -1/3
    c¹::Float64     # First-order correction c^(1) = -0.395 * B²
end

function default_parameters(; B = 0.5)
    @assert B < 0.6 "B should be kept smaller than 0.6 for accuracy"
    H = 0.40meters
    L = 295kilometers
    U = sqrt(g * H)  # [m/s] ≈ 1.981; ensures non-dim gravity wave speed = 1
    T = L / U        # [s]   ≈ 148,914 s ≈ 1.72 days
    return SolitonParameters(
        H, L, T, U,
        -24.0 * L,  24.0 * L,   # x domain [m]
         -8.0 * L,   8.0 * L,   # y domain [m]
        B,
        0.771 * B^2,    # A
        -1.0/3.0,       # c⁰
        -0.395 * B^2)   # c¹
end

# Hermite polynomial H_n(x) via recurrence
function hermite_polynomial(n::Int, x::Real)
    n == 0 && return 1.0
    n == 1 && return 2.0 * x
    H_nm2, H_nm1 = 1.0, 2.0 * x
    H_n = 0.0
    for i in 2:n
        H_n   = 2.0 * x * H_nm1 - 2.0 * (i - 1) * H_nm2
        H_nm2 = H_nm1
        H_nm1 = H_n
    end
    return H_n
end
hermite_polynomial(n::Int, x::AbstractArray) = hermite_polynomial.(n, x)

# Non-dimensional moving coordinate ξ = (x - c·t) / L, from dimensional inputs
function ξ_nd(x, t, p::SolitonParameters)
    c = (p.c⁰ + p.c¹) * p.U   # dimensional phase speed [m/s]
    return (x - c * t) / p.L
end

# Soliton envelope (non-dimensional): η_nd(ξ) = A · sech²(B·ξ)
envelope(ξ, p::SolitonParameters) = p.A * sech.(p.B * ξ).^2

# ∂envelope/∂ξ = -2B · tanh(B·ξ) · envelope  (non-dimensional)
∂envelope_∂ξ(ξ, e_val, p::SolitonParameters) = -2.0 * p.B * tanh.(p.B * ξ) .* e_val

# Zero-order solution (Boyd 1980) — operates on non-dimensional y and envelope
u⁰(y, e_val)  = e_val .* ((-9.0 .+ 6.0 * y.^2) / 4.0) .* exp.(-y.^2 / 2.0)
v⁰(y, de_val) = de_val .* (2.0 * y) .* exp.(-y.^2 / 2.0)
h⁰(y, e_val)  = e_val .* ((3.0 .+ 6.0 * y.^2) / 4.0) .* exp.(-y.^2 / 2.0)

# First-order Hermite series  f(y) = exp(-y²/2) · Σ fₙ · Hₙ(y)
function hermite_series(y, coeffs, n_max)
    result = zero(y)
    for n in 0:min(n_max, length(coeffs) - 1)
        result .+= coeffs[n+1] .* hermite_polynomial(n, y)
    end
    return exp.(-y.^2 / 2.0) .* result
end

# First-order corrections (Boyd 1985); Hermite coeffs from Table 6.1 optional
function u¹(y, e_val, p::SolitonParameters, u_coeffs = Float64[])
    term1 = p.c¹ .* e_val .* (9.0/16.0) .* (3.0 .+ 2.0 * y.^2) .* exp.(-y.^2 / 2.0)
    isempty(u_coeffs) && return term1
    return term1 .+ e_val.^2 .* hermite_series(y, u_coeffs, 10)
end

function v¹(y, de_val, e_val, v_coeffs = Float64[])
    isempty(v_coeffs) && return zero(y)
    return de_val .* e_val .* hermite_series(y, v_coeffs, 10)
end

function h¹(y, e_val, p::SolitonParameters, h_coeffs = Float64[])
    term1 = p.c¹ .* e_val .* (9.0/16.0) .* (-5.0 .+ 2.0 * y.^2) .* exp.(-y.^2 / 2.0)
    isempty(h_coeffs) && return term1
    return term1 .+ e_val.^2 .* hermite_series(y, h_coeffs, 10)
end

# Dimensional analytical functions: accept (x, y, t) in [m, m, s], return SI units
function analytic_u(x, y, t, p::SolitonParameters)
    ξ = ξ_nd(x, t, p)
    ŷ = y / p.L
    e_val = envelope(ξ, p)
    return (u⁰(ŷ, e_val) + u¹(ŷ, e_val, p)) * p.U   # [m/s]
end

function analytic_v(x, y, t, p::SolitonParameters)
    ξ = ξ_nd(x, t, p)
    ŷ = y / p.L
    e_val = envelope(ξ, p)
    return v⁰(ŷ, ∂envelope_∂ξ(ξ, e_val, p)) * p.U   # [m/s]
end

function analytic_η(x, y, t, p::SolitonParameters)
    ξ = ξ_nd(x, t, p)
    ŷ = y / p.L
    e_val = envelope(ξ, p)
    return (h⁰(ŷ, e_val) + h¹(ŷ, e_val, p)) * p.H   # [m]
end
#---

# Simulation setup
function setup_simulation(params::SolitonParameters;
                          Nx = 200, Ny = 100, Nz = 4,
                          stop_time = 70 * params.T,
                          scheme = PerturbationAdvection(),
                          ν_sponge = 1e4meters^2 / second, # peak viscosity
                          nudging_rate = 1 / (2days),
                          sponge_width = 1000kilometers,
                          outfile = joinpath("output", "hydrostatic_soliton.jld2"))

    # Grid
    z = MutableVerticalDiscretization(range(-params.H, 0, length = Nz + 1))

    grid = RectilinearGrid(CPU();
                           topology = (Bounded, Bounded, Bounded),
                           size = (Nx, Ny, Nz),
                           x = (params.x_min, params.x_max),
                           y = (params.y_min, params.y_max),
                           z,
                           halo = (8, 8, 8))

    # Model:
    β = params.U / params.L^2

    west_obc  = OpenBoundaryCondition((y, z, t) -> analytic_u(params.x_min, y, t, params); scheme)
    east_obc  = OpenBoundaryCondition((y, z, t) -> analytic_u(params.x_max, y, t, params); scheme)
    south_obc = OpenBoundaryCondition((x, z, t) -> analytic_v(x, params.y_min, t, params); scheme)
    north_obc = OpenBoundaryCondition((x, z, t) -> analytic_v(x, params.y_max, t, params); scheme)
    u_bcs = FieldBoundaryConditions(west = west_obc, east = east_obc)
    v_bcs = FieldBoundaryConditions(south = south_obc, north = north_obc)
    boundary_conditions = (; u = u_bcs, v = v_bcs)

    # Sponge layers: spatially varying horizontal viscosity that ramps from
    # ν_sponge at each boundary to zero at sponge_width inside the domain.
    west_mask  = PiecewiseLinearMask{:x}(center = params.x_min, width = sponge_width)
    east_mask  = PiecewiseLinearMask{:x}(center = params.x_max, width = sponge_width)
    south_mask = PiecewiseLinearMask{:y}(center = params.y_min, width = sponge_width)
    north_mask = PiecewiseLinearMask{:y}(center = params.y_max, width = sponge_width)

    @inline sponge_mask(x, y, z) = west_mask(x, y, z) + east_mask(x, y, z) + south_mask(x, y, z) + north_mask(x, y, z)
    @inline sponge_ν(x, y, z, t) = ν_sponge * sponge_mask(x, y, z)

    closure = HorizontalScalarDiffusivity(ν = sponge_ν)

    # Nudging: relax u and v toward the analytical solution in the sponge regions
    u_nudging = Relaxation(; rate = nudging_rate, mask = sponge_mask, target = (x, y, z, t) -> analytic_u(x, y, t, params))
    v_nudging = Relaxation(; rate = nudging_rate, mask = sponge_mask, target = (x, y, z, t) -> analytic_v(x, y, t, params))

    model = HydrostaticFreeSurfaceModel(grid;
        free_surface        = ImplicitFreeSurface(reltol = 1e-10, abstol = 1e-10, maxiter = 100,
                                                  gravitational_acceleration = g),
        momentum_advection  = WENO(order=5, minimum_buffer_upwind_order=1),
        timestepper = :SplitRungeKutta3,
        vertical_coordinate = ZStarCoordinate(),
        coriolis            = BetaPlane(f₀ = 0, β = β),
        closure,
        forcing             = (; u = u_nudging, v = v_nudging),
        boundary_conditions)

    # Initial conditions (soliton at t = 0)
    set!(model,
        u = (x, y, z) -> analytic_u(x, y, 0.0, params),
        v = (x, y, z) -> analytic_v(x, y, 0.0, params),
        η = (x, y, z) -> analytic_η(x, y, 0.0, params))

    # Simulation
    c_grav    = √(g * params.H) # surface gravity wave speed [m/s]
    max_Δt    = 0.5 * minimum_xspacing(grid) / c_grav # CFL limit from gravity waves
    Δt₀       = 0.1 * max_Δt
    simulation = Simulation(model; Δt = Δt₀, stop_time)
    conjure_time_step_wizard!(simulation, IterationInterval(10); cfl = 0.8, max_Δt)

    function progress_message(sim)
        u, v, w = sim.model.velocities
        η = sim.model.free_surface.displacement
        cg_solver = sim.model.free_surface.implicit_step_solver.preconditioned_conjugate_gradient_solver
        @printf("Iter: %5d, t: %s, Δt: %s, max|u|: %.3e m/s, max|η|: %.3e m, CG: %d\n",
                iteration(sim), prettytime(time(sim)), prettytime(sim.Δt),
                maximum(abs, u), maximum(abs, η), cg_solver.iteration)
    end
    simulation.callbacks[:progress] = Callback(progress_message, IterationInterval(50))

    # Output
    mkpath(dirname(outfile))
    u, v, w = model.velocities
    η = model.free_surface.displacement

    simulation.output_writers[:fields] = JLD2Writer(model,
        (; u, v, w, η);
        schedule           = TimeInterval(params.T),
        filename           = outfile,
        overwrite_existing = true)

    return simulation
end

#+++ Animation: numerical vs analytical
function plot_soliton(simulation::Simulation, params::SolitonParameters;
                      animation_file = nothing)

    outfile = simulation.output_writers[:fields].filepath
    isnothing(animation_file) && (animation_file = replace(outfile, ".jld2" => ".mp4"))
    grid    = simulation.model.grid
    Nz      = grid.Nz

    #+++ Load results
    u_ts = FieldTimeSeries(outfile, "u")
    v_ts = FieldTimeSeries(outfile, "v")
    η_ts = FieldTimeSeries(outfile, "η")
    times = u_ts.times
    @info "Loaded $(length(times)) snapshots from $outfile"
    #---

    #+++ Build figure
    # Coordinate arrays in km for plotting, in m for analytical evaluation
    xF_m  = nodes(grid, Face(),   Center(), Center())[1]
    xC_m  = nodes(grid, Center(), Center(), Center())[1]
    yC_m  = nodes(grid, Center(), Center(), Center())[2]
    yF_m  = nodes(grid, Center(), Face(),   Center())[2]
    xF_km = xF_m ./ 1e3;  xC_km = xC_m ./ 1e3
    yC_km = yC_m ./ 1e3;  yF_km = yF_m ./ 1e3

    # Colorbar limits fixed to initial analytical solution
    clim_u = maximum(abs, [analytic_u(x, y, 0.0, params) for x in xF_m, y in yC_m])
    clim_v = maximum(abs, [analytic_v(x, y, 0.0, params) for x in xC_m, y in yF_m])
    clim_η = maximum(abs, [analytic_η(x, y, 0.0, params) for x in xC_m, y in yC_m])

    n     = Observable(1)
    title = @lift @sprintf("Equatorial soliton — t = %.2f T  (%.1f days)",
                           times[$n] / params.T, times[$n] / 86400)

    num_u = @lift interior(u_ts[$n], :, :, Nz)
    num_v = @lift interior(v_ts[$n], :, :, Nz)
    num_η = @lift interior(η_ts[$n], :, :, 1)

    ana_u = @lift [analytic_u(x, y, times[$n], params) for x in xF_m, y in yC_m]
    ana_v = @lift [analytic_v(x, y, times[$n], params) for x in xC_m, y in yF_m]
    ana_η = @lift [analytic_η(x, y, times[$n], params) for x in xC_m, y in yC_m]

    fig = Figure(size = (1400, 900))
    Label(fig[0, 1:3], title, fontsize = 18, font = :bold)
    Label(fig[1, 1], "Numerical",  fontsize = 14, tellwidth = false)
    Label(fig[1, 2], "Analytical", fontsize = 14, tellwidth = false)

    ax_u_num = Axis(fig[2, 1], title = "u (surface) [m/s]", xlabel = "x [km]", ylabel = "y [km]")
    ax_u_ana = Axis(fig[2, 2], title = "u (analytical) [m/s]", xlabel = "x [km]", ylabel = "y [km]")
    hm_u_num = heatmap!(ax_u_num, xF_km, yC_km, num_u; colorrange = (-clim_u, clim_u), colormap = :balance)
    hm_u_ana = heatmap!(ax_u_ana, xF_km, yC_km, ana_u; colorrange = (-clim_u, clim_u), colormap = :balance)
    Colorbar(fig[2, 3], hm_u_num, label = "u [m/s]")

    ax_v_num = Axis(fig[3, 1], title = "v (surface) [m/s]", xlabel = "x [km]", ylabel = "y [km]")
    ax_v_ana = Axis(fig[3, 2], title = "v (analytical) [m/s]", xlabel = "x [km]", ylabel = "y [km]")
    hm_v_num = heatmap!(ax_v_num, xC_km, yF_km, num_v; colorrange = (-clim_v, clim_v), colormap = :balance)
    hm_v_ana = heatmap!(ax_v_ana, xC_km, yF_km, ana_v; colorrange = (-clim_v, clim_v), colormap = :balance)
    Colorbar(fig[3, 3], hm_v_num, label = "v [m/s]")

    ax_η_num = Axis(fig[4, 1], title = "η [m]", xlabel = "x [km]", ylabel = "y [km]")
    ax_η_ana = Axis(fig[4, 2], title = "η [m]", xlabel = "x [km]", ylabel = "y [km]")
    hm_η_num = heatmap!(ax_η_num, xC_km, yC_km, num_η; colorrange = (-clim_η, clim_η), colormap = :balance)
    hm_η_ana = heatmap!(ax_η_ana, xC_km, yC_km, ana_η; colorrange = (-clim_η, clim_η), colormap = :balance)
    Colorbar(fig[4, 3], hm_η_num, label = "η [m]")

    resize_to_layout!(fig)
    #---

    #+++ Record animation
    CairoMakie.record(fig, animation_file, 1:length(times), framerate = 12, compression = 20) do i
        n[] = i
    end
    @info "Animation saved to $animation_file"
    #---

    return fig
end
#---

params = default_parameters(B = 0.5)
simulation = setup_simulation(params; stop_time = 150days)
run!(simulation)
@info "Simulation complete."
plot_soliton(simulation, params)
