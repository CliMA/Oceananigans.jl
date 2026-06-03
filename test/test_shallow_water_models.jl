include("dependencies_for_runtests.jl")

using Oceananigans.Models.ShallowWaterModels
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary

function time_stepping_shallow_water_model_works(arch, topo, coriolis, advection; timestepper=:RungeKutta3)
    grid = RectilinearGrid(arch, size=(3, 3), extent=(2π, 2π), topology=topo)
    model = ShallowWaterModel(grid;
                              gravitational_acceleration = 1,
                              coriolis = coriolis,
                              momentum_advection = advection,
                              timestepper = :RungeKutta3)
    set!(model, h=1)

    simulation = Simulation(model, Δt=1.0, stop_iteration=1)
    run!(simulation)

    return model.clock.iteration == 1
end

function time_step_wizard_shallow_water_model_works(arch, topo, coriolis)
    grid = RectilinearGrid(arch, size=(3, 3), extent=(2π, 2π), topology=topo)
    model = ShallowWaterModel(grid;
                              gravitational_acceleration = 1,
                              coriolis = coriolis)
    set!(model, h=1)

    simulation = Simulation(model, Δt=1.0, stop_iteration=1)
    wizard = TimeStepWizard(cfl=1.0, max_change=1.1, max_Δt=10)
    simulation.callbacks[:wizard] = Callback(wizard)
    run!(simulation)

    return model.clock.iteration == 1
end

function shallow_water_model_tracers_and_forcings_work(arch)
    grid = RectilinearGrid(arch, size=(3, 3), extent=(2π, 2π), topology=(Periodic, Periodic, Flat))
    model = ShallowWaterModel(grid;
                              gravitational_acceleration = 1,
                              tracers = (:c, :d))
    set!(model, h=1)

    @test model.tracers.c isa Field
    @test model.tracers.d isa Field

    @test haskey(model.forcing, :uh)
    @test haskey(model.forcing, :vh)
    @test haskey(model.forcing, :h)
    @test haskey(model.forcing, :c)
    @test haskey(model.forcing, :d)

    simulation = Simulation(model, Δt=1.0, stop_iteration=1)
    run!(simulation)

    @test model.clock.iteration == 1

    return nothing
end

function test_shallow_water_diffusion_cosine(grid, formulation, fieldname, ξ)
    ν, m = 1, 2 # viscosity and cosine wavenumber

    closure = ShallowWaterScalarDiffusivity(; ν)
    momentum_advection = nothing
    tracer_advection = nothing
    mass_advection = nothing

    model = ShallowWaterModel(grid;
                              closure,
                              gravitational_acceleration = 1.0,
                              momentum_advection,
                              tracer_advection,
                              mass_advection,
                              formulation)

    field = model.velocities[fieldname]

    interior(field) .= on_architecture(architecture(grid), cos.(m * ξ))
    update_state!(model)

    # Step forward with small time-step relative to viscous/diffusive time scale
    Δt = 1e-6 * grid.Lx^2 / closure.ν
    for _ in 1:5
        time_step!(model, Δt)
    end

    diffusing_cosine(ξ, t, κ, m) = exp(-κ * m^2 * t) * cos(m * ξ)
    analytical_solution = Field(instantiated_location(field), grid)
    analytical_solution .= diffusing_cosine.(ξ, model.clock.time, ν, m)

    return isapprox(field, analytical_solution, atol=1e-6, rtol=1e-6)
end

function shallow_water_vector_invariant_octahealpix_tracer_transport_uses_nonorthogonal_fluxes(FT, N)
    grid = SphericalShellGrid(CPU(), FT;
                              size = (2N, 2N),
                              z = (zero(FT), one(FT)),
                              radius = one(FT),
                              topology = (Connected, Connected, Flat),
                              halo = (3, 3),
                              mapping = OctaHEALPixMapping(N))

    advection = Centered()
    formulation = VectorInvariantFormulation()

    model = ShallowWaterModel(grid;
                              gravitational_acceleration = one(FT),
                              tracers = (:c,),
                              tracer_advection = advection,
                              formulation = formulation)

    solution = model.solution
    c = model.tracers.c

    λu, φu, _ = nodes(grid, (Face(), Center(), Center()), reshape = true)
    λv, φv, _ = nodes(grid, (Center(), Face(), Center()), reshape = true)
    λc, φc, _ = nodes(grid, (Center(), Center(), Center()), reshape = true)

    interior(solution.u) .= @. one(FT) + convert(FT, 1//10) * sin(λu) + convert(FT, 1//12) * cos(φu)
    interior(solution.v) .= @. -convert(FT, 1//9) + convert(FT, 1//11) * cos(λv) - convert(FT, 1//13) * sin(φv)
    interior(solution.h) .= @. one(FT) + convert(FT, 1//8) * cos(λc) * cos(φc)
    interior(c)          .= @. convert(FT, 2//5) + convert(FT, 1//10) * sin(λc) - convert(FT, 1//14) * cos(φc)

    fill_halo_regions!(solution.u, solution.v)
    fill_halo_regions!(solution.h)
    fill_halo_regions!(c)

    Nx = grid.Nx
    Ny = grid.Ny

    actual_div_Uc = zeros(FT, Nx, Ny)
    expected_div_Uc = zeros(FT, Nx, Ny)
    naive_div_Uc = zeros(FT, Nx, Ny)

    actual_c_div_U = zeros(FT, Nx, Ny)
    expected_c_div_U = zeros(FT, Nx, Ny)
    naive_c_div_U = zeros(FT, Nx, Ny)
    converted_u = Oceananigans.Models.ShallowWaterModels.ShallowWaterConvertedTransportU(grid, solution)
    converted_v = Oceananigans.Models.ShallowWaterModels.ShallowWaterConvertedTransportV(grid, solution)

    for j in 1:Ny, i in 1:Nx
        actual_div_Uc[i, j] = Oceananigans.Models.ShallowWaterModels.div_Uc(i, j, 1, grid, advection, solution, c, formulation)
        expected_div_Uc[i, j] = 1 / Azᶜᶜᶜ(i, j, 1, grid) *
                                (δxᶜᵃᵃ(i, j, 1, grid, Oceananigans.Advection._advective_tracer_flux_x, advection, converted_u, c) +
                                 δyᵃᶜᵃ(i, j, 1, grid, Oceananigans.Advection._advective_tracer_flux_y, advection, converted_v, c))
        naive_div_Uc[i, j] = 1 / Azᶜᶜᶜ(i, j, 1, grid) *
                             (δxᶜᵃᵃ(i, j, 1, grid, Oceananigans.Advection._advective_tracer_flux_x, advection, solution[1], c) +
                              δyᵃᶜᵃ(i, j, 1, grid, Oceananigans.Advection._advective_tracer_flux_y, advection, solution[2], c))

        actual_c_div_U[i, j] = Oceananigans.Models.ShallowWaterModels.c_div_U(i, j, 1, grid, solution, c, formulation)
        expected_c_div_U[i, j] = c[i, j, 1] / Azᶜᶜᶜ(i, j, 1, grid) *
                                 Oceananigans.Operators.horizontal_volume_flux_div_xyᶜᶜᶜ(i, j, 1, grid, solution[1], solution[2])
        naive_c_div_U[i, j] = c[i, j, 1] / Azᶜᶜᶜ(i, j, 1, grid) *
                              (δxᶜᵃᵃ(i, j, 1, grid, Δy_qᶠᶜᶜ, solution[1]) +
                               δyᵃᶜᵃ(i, j, 1, grid, Δx_qᶜᶠᶜ, solution[2]))
    end

    div_Uc_scale = max(maximum(abs, expected_div_Uc), one(FT))
    c_div_U_scale = max(maximum(abs, expected_c_div_U), one(FT))

    @test isapprox(actual_div_Uc, expected_div_Uc; rtol = zero(FT), atol = 1000eps(FT) * div_Uc_scale)
    @test isapprox(actual_c_div_U, expected_c_div_U; rtol = zero(FT), atol = 1000eps(FT) * c_div_U_scale)

    @test maximum(abs, expected_div_Uc .- naive_div_Uc) > 100eps(FT) * div_Uc_scale
    @test maximum(abs, expected_c_div_U .- naive_c_div_U) > 100eps(FT) * c_div_U_scale

    return nothing
end

function shallow_water_vector_invariant_octahealpix_pressure_gradient_uses_covariant_gradient(FT, N)
    grid = SphericalShellGrid(CPU(), FT;
                              size = (2N, 2N),
                              z = (zero(FT), one(FT)),
                              radius = one(FT),
                              topology = (Connected, Connected, Flat),
                              halo = (3, 3),
                              mapping = OctaHEALPixMapping(N))

    formulation = VectorInvariantFormulation()

    model = ShallowWaterModel(grid;
                              gravitational_acceleration = one(FT),
                              momentum_advection = nothing,
                              formulation = formulation)

    solution = model.solution
    bathymetry = model.bathymetry
    g = model.gravitational_acceleration

    λc, φc, _ = nodes(grid, (Center(), Center(), Center()), reshape = true)

    interior(solution.h) .= @. one(FT) + convert(FT, 1//7) * sin(λc) - convert(FT, 1//9) * cos(φc)
    fill_halo_regions!(solution.h)

    Nx_u, Ny_u, _ = size(interior(solution.u))
    Nx_v, Ny_v, _ = size(interior(solution.v))

    actual_x = zeros(FT, Nx_u, Ny_u)
    expected_x = zeros(FT, Nx_u, Ny_u)
    naive_x = zeros(FT, Nx_u, Ny_u)

    actual_y = zeros(FT, Nx_v, Ny_v)
    expected_y = zeros(FT, Nx_v, Ny_v)
    naive_y = zeros(FT, Nx_v, Ny_v)

    for j in 1:Ny_u, i in 1:Nx_u
        actual_x[i, j] = Oceananigans.Models.ShallowWaterModels.x_pressure_gradient(i, j, 1, grid, g, solution.h, bathymetry, formulation)
        expected_x[i, j] = g * Oceananigans.Operators.covariant_gradient_xᶠᶜᶜ(i, j, 1, grid, Oceananigans.Models.ShallowWaterModels.h_plus_hB, solution.h, bathymetry)
        naive_x[i, j] = g * Oceananigans.Operators.δxᶠᶜᶜ(i, j, 1, grid,
                                                         Oceananigans.Models.ShallowWaterModels.h_plus_hB,
                                                         solution.h, bathymetry) *
                             Oceananigans.Operators.Δx⁻¹ᶠᶜᶜ(i, j, 1, grid)
    end

    for j in 1:Ny_v, i in 1:Nx_v
        actual_y[i, j] = Oceananigans.Models.ShallowWaterModels.y_pressure_gradient(i, j, 1, grid, g, solution.h, bathymetry, formulation)
        expected_y[i, j] = g * Oceananigans.Operators.covariant_gradient_yᶜᶠᶜ(i, j, 1, grid, Oceananigans.Models.ShallowWaterModels.h_plus_hB, solution.h, bathymetry)
        naive_y[i, j] = g * Oceananigans.Operators.δyᶜᶠᶜ(i, j, 1, grid,
                                                         Oceananigans.Models.ShallowWaterModels.h_plus_hB,
                                                         solution.h, bathymetry) *
                             Oceananigans.Operators.Δy⁻¹ᶜᶠᶜ(i, j, 1, grid)
    end

    x_scale = max(maximum(abs, expected_x), one(FT))
    y_scale = max(maximum(abs, expected_y), one(FT))

    @test isapprox(actual_x, expected_x; rtol = zero(FT), atol = 1000eps(FT) * x_scale)
    @test isapprox(actual_y, expected_y; rtol = zero(FT), atol = 1000eps(FT) * y_scale)

    @test maximum(abs, expected_x .- naive_x) > 100eps(FT) * x_scale
    @test maximum(abs, expected_y .- naive_y) > 100eps(FT) * y_scale

    return nothing
end

function shallow_water_octahealpix_cell_advection_timescale_uses_transport_scaling(FT, N)
    grid = SphericalShellGrid(CPU(), FT;
                              size = (2N, 2N),
                              z = (zero(FT), one(FT)),
                              radius = one(FT),
                              topology = (Connected, Connected, Flat),
                              halo = (3, 3),
                              mapping = OctaHEALPixMapping(N))

    model = ShallowWaterModel(grid;
                              gravitational_acceleration = one(FT),
                              momentum_advection = nothing,
                              formulation = VectorInvariantFormulation())

    solution = model.solution

    λu, φu, _ = nodes(grid, (Face(), Center(), Center()), reshape = true)
    λv, φv, _ = nodes(grid, (Center(), Face(), Center()), reshape = true)

    interior(solution.u) .= @. one(FT) + convert(FT, 1//10) * sin(λu) + convert(FT, 1//12) * cos(φu)
    interior(solution.v) .= @. -convert(FT, 1//9) + convert(FT, 1//11) * cos(λv) - convert(FT, 1//13) * sin(φv)

    fill_halo_regions!(solution.u, solution.v)

    actual_τ = Oceananigans.Advection.cell_advection_timescale(model)

    expected_τ = typemax(FT)
    naive_τ = typemax(FT)

    for j in 1:grid.Ny, i in 1:grid.Nx
        transport_u = Oceananigans.Operators.covariant_to_volume_flux_uᶠᶜᶜ(i, j, 1, grid, solution.u, solution.v)
        transport_v = Oceananigans.Operators.covariant_to_volume_flux_vᶜᶠᶜ(i, j, 1, grid, solution.u, solution.v)
        Az = Azᶜᶜᶜ(i, j, 1, grid)

        expected_τ = min(expected_τ, min(Az / abs(transport_u), Az / abs(transport_v)))
        naive_τ = min(naive_τ,
                      min(Δxᶠᶜᶜ(i, j, 1, grid) / abs(solution.u[i, j, 1]),
                          Δyᶜᶠᶜ(i, j, 1, grid) / abs(solution.v[i, j, 1])))
    end

    scale = max(abs(expected_τ), one(FT))

    @test isapprox(actual_τ, expected_τ; rtol = zero(FT), atol = 1000eps(FT) * scale)
    @test abs(expected_τ - naive_τ) > 100eps(FT) * scale

    return nothing
end

function shallow_water_conservative_octahealpix_model_update_uses_paired_compute(FT, N)
    grid = SphericalShellGrid(CPU(), FT;
                              size = (2N, 2N),
                              z = (zero(FT), one(FT)),
                              radius = one(FT),
                              topology = (Connected, Connected, Flat),
                              halo = (3, 3),
                              mapping = OctaHEALPixMapping(N))

    model = ShallowWaterModel(grid;
                              gravitational_acceleration = one(FT),
                              tracers = (:c,),
                              momentum_advection = Centered(),
                              tracer_advection = Centered(),
                              mass_advection = Centered(),
                              formulation = ConservativeFormulation())

    solution = model.solution
    c = model.tracers.c

    λuh, φuh, _ = nodes(grid, (Face(), Center(), Center()), reshape = true)
    λvh, φvh, _ = nodes(grid, (Center(), Face(), Center()), reshape = true)
    λc, φc, _ = nodes(grid, (Center(), Center(), Center()), reshape = true)

    interior(solution.uh) .= @. one(FT) + convert(FT, 1//10) * sin(λuh) + convert(FT, 1//12) * cos(φuh)
    interior(solution.vh) .= @. convert(FT, 2) - convert(FT, 1//11) * cos(λvh) + convert(FT, 1//13) * sin(φvh)
    interior(solution.h)  .= @. one(FT) + convert(FT, 1//8) * cos(λc) * cos(φc)
    interior(c)           .= @. convert(FT, 2//5) + convert(FT, 1//10) * sin(λc) - convert(FT, 1//14) * cos(φc)

    update_state!(model)

    expected_u = Field(solution.uh / solution.h;
                       boundary_conditions = model.velocities.u.boundary_conditions,
                       compute = false)
    expected_v = Field(solution.vh / solution.h;
                       boundary_conditions = model.velocities.v.boundary_conditions,
                       compute = false)
    Oceananigans.Fields.compute!((expected_u, expected_v))

    u_error = zero(FT)
    v_error = zero(FT)

    for k in 1:grid.Nz, j in 0:(grid.Ny + 1), i in 0:(grid.Nx + 1)
        u_error = max(u_error, abs(model.velocities.u[i, j, k] - expected_u[i, j, k]))
        v_error = max(v_error, abs(model.velocities.v[i, j, k] - expected_v[i, j, k]))
    end

    @test u_error == zero(FT)
    @test v_error == zero(FT)

    return nothing
end

function shallow_water_conservative_octahealpix_time_step_works(FT, N)
    grid = SphericalShellGrid(CPU(), FT;
                              size = (2N, 2N),
                              z = (zero(FT), one(FT)),
                              radius = one(FT),
                              topology = (Connected, Connected, Flat),
                              halo = (3, 3),
                              mapping = OctaHEALPixMapping(N))

    model = ShallowWaterModel(grid;
                              gravitational_acceleration = one(FT),
                              tracers = :c,
                              momentum_advection = Centered(FT),
                              tracer_advection = Centered(FT),
                              mass_advection = Centered(FT),
                              formulation = ConservativeFormulation())

    solution = model.solution
    c = model.tracers.c

    λuh, φuh, _ = nodes(grid, (Face(), Center(), Center()), reshape = true)
    λvh, φvh, _ = nodes(grid, (Center(), Face(), Center()), reshape = true)
    λc, φc, _ = nodes(grid, (Center(), Center(), Center()), reshape = true)

    interior(solution.uh) .= @. one(FT) + convert(FT, 1//10) * sin(λuh) + convert(FT, 1//12) * cos(φuh)
    interior(solution.vh) .= @. convert(FT, 2) - convert(FT, 1//11) * cos(λvh) + convert(FT, 1//13) * sin(φvh)
    interior(solution.h)  .= @. one(FT) + convert(FT, 1//8) * cos(λc) * cos(φc)
    interior(c)           .= @. convert(FT, 2//5) + convert(FT, 1//10) * sin(λc) - convert(FT, 1//14) * cos(φc)

    fill_halo_regions!((solution.uh, solution.vh))
    fill_halo_regions!(solution.h)
    fill_halo_regions!(c)

    uh₀ = Array(interior(solution.uh))
    vh₀ = Array(interior(solution.vh))
    h₀ = Array(interior(solution.h))
    c₀ = Array(interior(c))

    Δt = convert(FT, 1//200)

    for _ in 1:5
        time_step!(model, Δt)
    end

    uh₁ = Array(interior(solution.uh))
    vh₁ = Array(interior(solution.vh))
    h₁ = Array(interior(solution.h))
    c₁ = Array(interior(c))

    @test all(isfinite, uh₁)
    @test all(isfinite, vh₁)
    @test all(isfinite, h₁)
    @test all(isfinite, c₁)
    @test minimum(h₁) > zero(FT)

    @test maximum(abs.(uh₁ .- uh₀)) > 100eps(FT)
    @test maximum(abs.(vh₁ .- vh₀)) > 100eps(FT)
    @test maximum(abs.(h₁ .- h₀)) > 100eps(FT)
    @test maximum(abs.(c₁ .- c₀)) > 100eps(FT)

    expected_u = Field(solution.uh / solution.h;
                       boundary_conditions = model.velocities.u.boundary_conditions,
                       compute = false)
    expected_v = Field(solution.vh / solution.h;
                       boundary_conditions = model.velocities.v.boundary_conditions,
                       compute = false)
    Oceananigans.Fields.compute!((expected_u, expected_v))

    u_error = zero(FT)
    v_error = zero(FT)

    for k in 1:grid.Nz, j in 0:(grid.Ny + 1), i in 0:(grid.Nx + 1)
        u_error = max(u_error, abs(model.velocities.u[i, j, k] - expected_u[i, j, k]))
        v_error = max(v_error, abs(model.velocities.v[i, j, k] - expected_v[i, j, k]))
    end

    @test u_error == zero(FT)
    @test v_error == zero(FT)

    return nothing
end

function shallow_water_conservative_octahealpix_closure_field_refreshes_halos(FT, N)
    grid = SphericalShellGrid(CPU(), FT;
                              size = (2N, 2N),
                              z = (zero(FT), one(FT)),
                              radius = one(FT),
                              topology = (Connected, Connected, Flat),
                              halo = (3, 3),
                              mapping = OctaHEALPixMapping(N))

    closure = ShallowWaterScalarDiffusivity(FT; ν = convert(FT, 1//5))

    function build_model()
        model = ShallowWaterModel(grid;
                                  closure,
                                  gravitational_acceleration = one(FT),
                                  momentum_advection = Centered(FT),
                                  tracer_advection = nothing,
                                  mass_advection = Centered(FT),
                                  formulation = ConservativeFormulation())

        solution = model.solution

        λuh, φuh, _ = nodes(grid, (Face(), Center(), Center()), reshape = true)
        λvh, φvh, _ = nodes(grid, (Center(), Face(), Center()), reshape = true)
        λc, φc, _ = nodes(grid, (Center(), Center(), Center()), reshape = true)

        interior(solution.uh) .= @. one(FT) + convert(FT, 1//10) * sin(λuh) + convert(FT, 1//12) * cos(φuh)
        interior(solution.vh) .= @. convert(FT, 2) - convert(FT, 1//11) * cos(λvh) + convert(FT, 1//13) * sin(φvh)
        interior(solution.h)  .= @. one(FT) + convert(FT, 1//8) * cos(λc) * cos(φc)

        return model
    end

    function corrupt_closure_halos!(field)
        for k in 1:grid.Nz, j in 0:(grid.Ny + 1), i in 0:(grid.Nx + 1)
            (1 <= i <= grid.Nx && 1 <= j <= grid.Ny) && continue
            field[i, j, k] = convert(FT, 321)
        end

        return nothing
    end

    reference_model = build_model()
    Oceananigans.Models.ShallowWaterModels.refresh_shallow_water_auxiliary_state!(reference_model)

    model = build_model()
    corrupt_closure_halos!(model.closure_fields.νₑ)
    Oceananigans.Models.ShallowWaterModels.refresh_shallow_water_auxiliary_state!(model)

    ν_error = zero(FT)

    for k in 1:grid.Nz, j in 0:(grid.Ny + 1), i in 0:(grid.Nx + 1)
        ν_error = max(ν_error,
                      abs(model.closure_fields.νₑ[i, j, k] -
                          reference_model.closure_fields.νₑ[i, j, k]))
    end

    @test ν_error == zero(FT)

    return nothing
end

function shallow_water_conservative_octahealpix_tracer_tendency_uses_velocity_divergence(FT, N)
    grid = SphericalShellGrid(CPU(), FT;
                              size = (2N, 2N),
                              z = (zero(FT), one(FT)),
                              radius = one(FT),
                              topology = (Connected, Connected, Flat),
                              halo = (3, 3),
                              mapping = OctaHEALPixMapping(N))

    model = ShallowWaterModel(grid;
                              gravitational_acceleration = one(FT),
                              tracers = :c,
                              momentum_advection = nothing,
                              tracer_advection = Centered(),
                              mass_advection = Centered(),
                              formulation = ConservativeFormulation())

    solution = model.solution
    c = model.tracers.c

    λuh, φuh, _ = nodes(grid, (Face(), Center(), Center()), reshape = true)
    λvh, φvh, _ = nodes(grid, (Center(), Face(), Center()), reshape = true)
    λc, φc, _ = nodes(grid, (Center(), Center(), Center()), reshape = true)

    interior(solution.uh) .= @. one(FT) + convert(FT, 1//10) * sin(λuh) + convert(FT, 1//12) * cos(φuh)
    interior(solution.vh) .= @. convert(FT, 2//3) - convert(FT, 1//11) * cos(λvh) + convert(FT, 1//13) * sin(φvh)
    interior(solution.h)  .= @. one(FT) + convert(FT, 1//8) * cos(λc) * cos(φc)
    interior(c)           .= @. convert(FT, 2//5) + convert(FT, 1//10) * sin(λc) - convert(FT, 1//14) * cos(φc)

    fill_halo_regions!((solution.uh, solution.vh))
    fill_halo_regions!(solution.h)
    fill_halo_regions!(c)

    compute_tendencies!(model, ())

    actual_c_tendency = Array(interior(model.timestepper.Gⁿ[4]))
    expected_c_tendency = zeros(FT, grid.Nx, grid.Ny)
    naive_c_tendency = zeros(FT, grid.Nx, grid.Ny)

    advection = model.advection.c
    conservative_velocity_u =
        Oceananigans.Models.ShallowWaterModels.ConservativeShallowWaterVelocityU(grid, solution)
    conservative_velocity_v =
        Oceananigans.Models.ShallowWaterModels.ConservativeShallowWaterVelocityV(grid, solution)

    for j in 1:grid.Ny, i in 1:grid.Nx
        Az = Azᶜᶜᶜ(i, j, 1, grid)

        div_Uc = one(FT) / Az *
                 (δxᶜᵃᵃ(i, j, 1, grid,
                      Oceananigans.Models.ShallowWaterModels.transport_tracer_flux_x,
                      advection,
                      solution.uh,
                      solution.h,
                      c) +
                  δyᵃᶜᵃ(i, j, 1, grid,
                      Oceananigans.Models.ShallowWaterModels.transport_tracer_flux_y,
                      advection,
                      solution.vh,
                      solution.h,
                      c))

        conservative_velocity_divergence =
            one(FT) / Az *
            Oceananigans.Operators.horizontal_volume_flux_div_xyᶜᶜᶜ(i, j, 1, grid,
                                                                    conservative_velocity_u,
                                                                    conservative_velocity_v)

        naive_transport_divergence =
            one(FT) / Az *
            Oceananigans.Operators.horizontal_volume_flux_div_xyᶜᶜᶜ(i, j, 1, grid,
                                                                    solution.uh,
                                                                    solution.vh)

        expected_c_tendency[i, j] = -div_Uc + c[i, j, 1] * conservative_velocity_divergence
        naive_c_tendency[i, j] = -div_Uc + c[i, j, 1] * naive_transport_divergence
    end

    scale = max(maximum(abs, expected_c_tendency), one(FT))

    @test isapprox(actual_c_tendency, expected_c_tendency; rtol = zero(FT), atol = 1000eps(FT) * scale)
    @test maximum(abs, expected_c_tendency .- naive_c_tendency) > 100eps(FT) * scale

    return nothing
end

function shallow_water_octahealpix_bounds_preserving_tracer_time_step_works(FT, N)
    grid = SphericalShellGrid(CPU(), FT;
                              size = (2N, 2N),
                              z = (zero(FT), one(FT)),
                              radius = one(FT),
                              topology = (Connected, Connected, Flat),
                              halo = (3, 3),
                              mapping = OctaHEALPixMapping(N))

    model = ShallowWaterModel(grid;
                              gravitational_acceleration = one(FT),
                              tracers = :c,
                              momentum_advection = nothing,
                              mass_advection = nothing,
                              tracer_advection = WENO(FT; order = 1, bounds = (zero(FT), one(FT))),
                              formulation = VectorInvariantFormulation())

    solution = model.solution
    c = model.tracers.c

    λu, φu, _ = nodes(grid, (Face(), Center(), Center()), reshape = true)
    λv, φv, _ = nodes(grid, (Center(), Face(), Center()), reshape = true)
    λc, φc, _ = nodes(grid, (Center(), Center(), Center()), reshape = true)

    interior(solution.u) .= @. convert(FT, 1//6) + convert(FT, 1//12) * sin(λu) + convert(FT, 1//14) * cos(φu)
    interior(solution.v) .= @. -convert(FT, 1//7) + convert(FT, 1//13) * cos(λv) - convert(FT, 1//15) * sin(φv)
    interior(solution.h) .= one(FT)
    interior(c)          .= @. convert(FT, 1//2) + convert(FT, 1//4) * sin(λc) * cos(φc)

    fill_halo_regions!(solution.u, solution.v)
    fill_halo_regions!(solution.h)
    fill_halo_regions!(c)

    c₀ = Array(interior(c))

    Δt = convert(FT, 1//100)

    for _ in 1:5
        time_step!(model, Δt)
    end

    c₁ = Array(interior(c))

    @test all(isfinite, c₁)
    @test minimum(c₁) ≥ -1000eps(FT)
    @test maximum(c₁) ≤ one(FT) + 1000eps(FT)
    @test maximum(abs.(c₁ .- c₀)) > 100eps(FT)

    return nothing
end

function shallow_water_octahealpix_mass_tendency_uses_nonorthogonal_transport(FT, N)
    grid = SphericalShellGrid(CPU(), FT;
                              size = (2N, 2N),
                              z = (zero(FT), one(FT)),
                              radius = one(FT),
                              topology = (Connected, Connected, Flat),
                              halo = (3, 3),
                              mapping = OctaHEALPixMapping(N))

    model = ShallowWaterModel(grid;
                              gravitational_acceleration = one(FT),
                              momentum_advection = nothing,
                              tracer_advection = nothing,
                              mass_advection = Centered(),
                              formulation = VectorInvariantFormulation())

    solution = model.solution

    λu, φu, _ = nodes(grid, (Face(), Center(), Center()), reshape = true)
    λv, φv, _ = nodes(grid, (Center(), Face(), Center()), reshape = true)
    λc, φc, _ = nodes(grid, (Center(), Center(), Center()), reshape = true)

    interior(solution.u) .= @. convert(FT, 1//5) + convert(FT, 1//11) * sin(λu) + convert(FT, 1//13) * cos(φu)
    interior(solution.v) .= @. -convert(FT, 1//6) + convert(FT, 1//12) * cos(λv) - convert(FT, 1//14) * sin(φv)
    interior(solution.h) .= @. one(FT) + convert(FT, 1//9) * sin(λc) - convert(FT, 1//10) * cos(φc)

    fill_halo_regions!(solution.u, solution.v)
    fill_halo_regions!(solution.h)

    compute_tendencies!(model, ())

    actual_h_tendency = Array(interior(model.timestepper.Gⁿ[3]))

    expected_h_tendency = zeros(FT, grid.Nx, grid.Ny)
    naive_h_tendency = zeros(FT, grid.Nx, grid.Ny)
    converted_u = Oceananigans.Models.ShallowWaterModels.ShallowWaterConvertedTransportU(grid, solution)
    converted_v = Oceananigans.Models.ShallowWaterModels.ShallowWaterConvertedTransportV(grid, solution)

    for j in 1:grid.Ny, i in 1:grid.Nx
        Az = Oceananigans.Operators.Azᶜᶜᶜ(i, j, 1, grid)

        expected_h_tendency[i, j] = -one(FT) / Az *
                                    (Oceananigans.Operators.δxᶜᵃᵃ(i, j, 1, grid,
                                                                 Oceananigans.Advection._advective_tracer_flux_x,
                                                                 model.advection.mass,
                                                                 converted_u,
                                                                 solution.h) +
                                     Oceananigans.Operators.δyᵃᶜᵃ(i, j, 1, grid,
                                                                 Oceananigans.Advection._advective_tracer_flux_y,
                                                                 model.advection.mass,
                                                                 converted_v,
                                                                 solution.h))

        naive_h_tendency[i, j] = -one(FT) / Az *
                                 (Oceananigans.Operators.δxᶜᵃᵃ(i, j, 1, grid,
                                                              Oceananigans.Advection._advective_tracer_flux_x,
                                                              model.advection.mass,
                                                              solution.u,
                                                              solution.h) +
                                  Oceananigans.Operators.δyᵃᶜᵃ(i, j, 1, grid,
                                                              Oceananigans.Advection._advective_tracer_flux_y,
                                                              model.advection.mass,
                                                              solution.v,
                                                              solution.h))
    end

    scale = max(maximum(abs, expected_h_tendency), one(FT))

    @test isapprox(actual_h_tendency, expected_h_tendency; rtol = zero(FT), atol = 1000eps(FT) * scale)
    @test maximum(abs, expected_h_tendency .- naive_h_tendency) > 100eps(FT) * scale

    return nothing
end

function shallow_water_octahealpix_vector_invariant_momentum_tendencies_match_live_advection(FT, N)
    grid = SphericalShellGrid(CPU(), FT;
                              size = (2N, 2N),
                              z = (zero(FT), one(FT)),
                              radius = one(FT),
                              topology = (Connected, Connected, Flat),
                              halo = (3, 3),
                              mapping = OctaHEALPixMapping(N))

    momentum_advection = VectorInvariant(FT;
                                         vorticity_scheme = Oceananigans.Advection.EnergyConserving(FT),
                                         vertical_advection_scheme = Oceananigans.Advection.EnergyConserving(FT),
                                         kinetic_energy_gradient_scheme = Oceananigans.Advection.EnergyConserving(FT),
                                         divergence_scheme = Oceananigans.Advection.EnergyConserving(FT))

    formulation = VectorInvariantFormulation()

    model = ShallowWaterModel(grid;
                              gravitational_acceleration = one(FT),
                              momentum_advection = momentum_advection,
                              mass_advection = nothing,
                              formulation = formulation)

    solution = model.solution

    λu, φu, _ = nodes(grid, (Face(), Center(), Center()), reshape = true)
    λv, φv, _ = nodes(grid, (Center(), Face(), Center()), reshape = true)

    interior(solution.u) .= @. convert(FT, 1//8) + convert(FT, 1//10) * sin(λu) + convert(FT, 1//12) * cos(φu)
    interior(solution.v) .= @. -convert(FT, 1//9) + convert(FT, 1//11) * cos(λv) - convert(FT, 1//13) * sin(φv)
    interior(solution.h) .= one(FT)

    fill_halo_regions!(solution.u, solution.v)
    fill_halo_regions!(solution.h)

    Oceananigans.TimeSteppers.compute_tendencies!(model, ())

    Gⁿu = Array(interior(model.timestepper.Gⁿ[1]))
    Gⁿv = Array(interior(model.timestepper.Gⁿ[2]))
    Gⁿh = Array(interior(model.timestepper.Gⁿ[3]))

    expected_u = zeros(FT, size(Gⁿu)...)
    expected_v = zeros(FT, size(Gⁿv)...)

    for j in 1:size(Gⁿu, 2), i in 1:size(Gⁿu, 1)
        expected_u[i, j] = -Oceananigans.Models.ShallowWaterModels.div_mom_u(i, j, 1, grid, momentum_advection, solution, formulation)
    end

    for j in 1:size(Gⁿv, 2), i in 1:size(Gⁿv, 1)
        expected_v[i, j] = -Oceananigans.Models.ShallowWaterModels.div_mom_v(i, j, 1, grid, momentum_advection, solution, formulation)
    end

    u_scale = max(maximum(abs, expected_u), one(FT))
    v_scale = max(maximum(abs, expected_v), one(FT))

    @test isapprox(Gⁿu, expected_u; rtol = zero(FT), atol = 1000eps(FT) * u_scale)
    @test isapprox(Gⁿv, expected_v; rtol = zero(FT), atol = 1000eps(FT) * v_scale)
    @test maximum(abs, Gⁿh) ≤ 1000eps(FT)

    return nothing
end

function shallow_water_octahealpix_ab2_step_updates_vector_invariant_momentum(FT, N)
    grid = SphericalShellGrid(CPU(), FT;
                              size = (2N, 2N),
                              z = (zero(FT), one(FT)),
                              radius = one(FT),
                              topology = (Connected, Connected, Flat),
                              halo = (3, 3),
                              mapping = OctaHEALPixMapping(N))

    momentum_advection = VectorInvariant(FT;
                                         vorticity_scheme = Oceananigans.Advection.EnergyConserving(FT),
                                         vertical_advection_scheme = Oceananigans.Advection.EnergyConserving(FT),
                                         kinetic_energy_gradient_scheme = Oceananigans.Advection.EnergyConserving(FT),
                                         divergence_scheme = Oceananigans.Advection.EnergyConserving(FT))

    formulation = VectorInvariantFormulation()

    model = ShallowWaterModel(grid;
                              gravitational_acceleration = one(FT),
                              momentum_advection = momentum_advection,
                              mass_advection = nothing,
                              timestepper = :QuasiAdamsBashforth2,
                              formulation = formulation)

    solution = model.solution

    λu, φu, _ = nodes(grid, (Face(), Center(), Center()), reshape = true)
    λv, φv, _ = nodes(grid, (Center(), Face(), Center()), reshape = true)

    interior(solution.u) .= @. convert(FT, 1//8) + convert(FT, 1//10) * sin(λu) + convert(FT, 1//12) * cos(φu)
    interior(solution.v) .= @. -convert(FT, 1//9) + convert(FT, 1//11) * cos(λv) - convert(FT, 1//13) * sin(φv)
    interior(solution.h) .= one(FT)

    fill_halo_regions!(solution.u, solution.v)
    fill_halo_regions!(solution.h)

    u₀ = Array(interior(solution.u))
    v₀ = Array(interior(solution.v))
    h₀ = Array(interior(solution.h))

    parent(model.timestepper.G⁻[1]) .= zero(FT)
    parent(model.timestepper.G⁻[2]) .= zero(FT)
    parent(model.timestepper.G⁻[3]) .= zero(FT)

    Oceananigans.TimeSteppers.compute_tendencies!(model, ())

    Gⁿu = Array(interior(model.timestepper.Gⁿ[1]))
    Gⁿv = Array(interior(model.timestepper.Gⁿ[2]))
    Gⁿh = Array(interior(model.timestepper.Gⁿ[3]))

    χ = model.timestepper.χ
    Δt = convert(FT, 1//100)
    coefficient = convert(FT, 3//2) + χ

    expected_u = @. u₀ + Δt * coefficient * Gⁿu
    expected_v = @. v₀ + Δt * coefficient * Gⁿv
    expected_h = @. h₀ + Δt * coefficient * Gⁿh

    Oceananigans.TimeSteppers.ab2_step!(model, Δt, ())

    u₁ = Array(interior(solution.u))
    v₁ = Array(interior(solution.v))
    h₁ = Array(interior(solution.h))

    u_scale = max(maximum(abs, expected_u), one(FT))
    v_scale = max(maximum(abs, expected_v), one(FT))
    h_scale = max(maximum(abs, expected_h), one(FT))

    @test isapprox(u₁, expected_u; rtol = zero(FT), atol = 1000eps(FT) * u_scale)
    @test isapprox(v₁, expected_v; rtol = zero(FT), atol = 1000eps(FT) * v_scale)
    @test isapprox(h₁, expected_h; rtol = zero(FT), atol = 1000eps(FT) * h_scale)

    return nothing
end

function shallow_water_octahealpix_multistep_end_to_end_conserves_mass_and_tracer_mass(FT, N)
    grid = SphericalShellGrid(CPU(), FT;
                              size = (2N, 2N),
                              z = (zero(FT), one(FT)),
                              radius = one(FT),
                              topology = (Connected, Connected, Flat),
                              halo = (3, 3),
                              mapping = OctaHEALPixMapping(N))

    momentum_advection = VectorInvariant(FT;
                                         vorticity_scheme = Oceananigans.Advection.EnergyConserving(FT),
                                         vertical_advection_scheme = Oceananigans.Advection.EnergyConserving(FT),
                                         kinetic_energy_gradient_scheme = Oceananigans.Advection.EnergyConserving(FT),
                                         divergence_scheme = Oceananigans.Advection.EnergyConserving(FT))

    model = ShallowWaterModel(grid;
                              gravitational_acceleration = one(FT),
                              tracers = :c,
                              momentum_advection = momentum_advection,
                              mass_advection = Centered(FT; order = 2),
                              tracer_advection = Centered(FT; order = 2),
                              formulation = VectorInvariantFormulation())

    solution = model.solution
    c = model.tracers.c

    λu, φu, _ = nodes(grid, (Face(), Center(), Center()), reshape = true)
    λv, φv, _ = nodes(grid, (Center(), Face(), Center()), reshape = true)
    λc, φc, _ = nodes(grid, (Center(), Center(), Center()), reshape = true)

    interior(solution.u) .= @. convert(FT, 1//20) + convert(FT, 1//30) * sin(λu) + convert(FT, 1//35) * cos(φu)
    interior(solution.v) .= @. -convert(FT, 1//22) + convert(FT, 1//32) * cos(λv) - convert(FT, 1//36) * sin(φv)
    interior(solution.h) .= @. one(FT) + convert(FT, 1//12) * sin(λc) - convert(FT, 1//15) * cos(φc)
    interior(c)          .= @. convert(FT, 1//2) + convert(FT, 1//8) * sin(λc) * cos(φc)

    fill_halo_regions!(solution.u, solution.v)
    fill_halo_regions!(solution.h)
    fill_halo_regions!(c)

    function total_mass(field)
        total = zero(FT)

        for j in 1:grid.Ny, i in 1:grid.Nx
            total += Azᶜᶜᶜ(i, j, 1, grid) * field[i, j, 1]
        end

        return total
    end

    function total_tracer_mass(field, tracer)
        total = zero(FT)

        for j in 1:grid.Ny, i in 1:grid.Nx
            total += Azᶜᶜᶜ(i, j, 1, grid) * field[i, j, 1] * tracer[i, j, 1]
        end

        return total
    end

    u₀ = Array(interior(solution.u))
    v₀ = Array(interior(solution.v))
    h₀ = Array(interior(solution.h))
    c₀ = Array(interior(c))

    M₀ = total_mass(solution.h)
    C₀ = total_tracer_mass(solution.h, c)

    Δt = convert(FT, 1//200)

    for _ in 1:5
        time_step!(model, Δt)
    end

    u₁ = Array(interior(solution.u))
    v₁ = Array(interior(solution.v))
    h₁ = Array(interior(solution.h))
    c₁ = Array(interior(c))

    M₁ = total_mass(solution.h)
    C₁ = total_tracer_mass(solution.h, c)

    mass_scale = max(abs(M₀), one(FT))
    tracer_mass_scale = max(abs(C₀), one(FT))

    @test all(isfinite, u₁)
    @test all(isfinite, v₁)
    @test all(isfinite, h₁)
    @test all(isfinite, c₁)
    @test minimum(h₁) > zero(FT)

    @test maximum(abs.(u₁ .- u₀)) > 100eps(FT)
    @test maximum(abs.(v₁ .- v₀)) > 100eps(FT)
    @test maximum(abs.(h₁ .- h₀)) > 100eps(FT)
    @test maximum(abs.(c₁ .- c₀)) > 100eps(FT)

    @test isapprox(M₁, M₀; rtol = zero(FT), atol = 10000eps(FT) * mass_scale)
    @test isapprox(C₁, C₀; rtol = zero(FT), atol = 10000eps(FT) * tracer_mass_scale)

    return nothing
end

@testset "Shallow Water Models" begin
    @info "Testing shallow water models..."

    @testset "Must be Flat in the vertical" begin
        grid = RectilinearGrid(size=(1, 1, 1), extent=(1, 1, 1), topology=(Periodic, Periodic, Bounded))
        @test_throws ArgumentError ShallowWaterModel(grid; gravitational_acceleration=1)

        grid = RectilinearGrid(size=(1, 1, 1), extent=(1, 1, 1), topology=(Periodic, Periodic, Periodic))
        @test_throws ArgumentError ShallowWaterModel(grid; gravitational_acceleration=1)
    end

    @testset "Model constructor errors" begin
        grid = RectilinearGrid(size=(1, 1), extent=(1, 1), topology=(Periodic,Periodic,Flat))
        @test_throws ArgumentError ShallowWaterModel(grid; gravitational_acceleration=1)
        @test_throws ArgumentError ShallowWaterModel(grid; gravitational_acceleration=1)
    end

    topo = (Flat, Flat, Flat)

    @testset "$topo model construction" begin
    @info "  Testing $topo model construction..."
        for arch in archs, FT in float_types
            grid = RectilinearGrid(arch, FT, topology=topo, size=(), extent=())
            model = ShallowWaterModel(grid; gravitational_acceleration=1)

            @test model isa ShallowWaterModel
        end
    end

    topos = (
             (Bounded, Flat,    Flat),
             (Flat,    Bounded, Flat),
            )

    for topo in topos
        @testset "$topo model construction" begin
            @info "  Testing $topo model construction..."
            for arch in archs, FT in float_types
                #arch isa GPU && topo == (Flat, Bounded, Flat) && continue

                grid = RectilinearGrid(arch, FT, topology=topo, size=3, extent=1, halo=3)
                model = ShallowWaterModel(grid; gravitational_acceleration=1)

                @test model isa ShallowWaterModel
            end
        end
    end

    topos = (
             (Periodic, Periodic, Flat),
             (Periodic,  Bounded, Flat),
             (Bounded,   Bounded, Flat),
            )

    for topo in topos
        @testset "$topo model construction" begin
            @info "  Testing $topo model construction..."
            for arch in archs, FT in float_types
               #arch isa GPU && topo == (Bounded, Bounded, Flat) && continue

                grid = RectilinearGrid(arch, FT, topology=topo, size=(3, 3), extent=(1, 2), halo=(3, 3))
                model = ShallowWaterModel(grid; gravitational_acceleration=1)

                @test model isa ShallowWaterModel
            end
        end
    end

    @testset "Setting ShallowWaterModel fields" begin
        @info "  Testing setting shallow water model fields..."

        for arch in archs, FT in float_types
            N = (4,   4)
            L = (2π, 3π)

            grid = RectilinearGrid(arch, FT, size=N, extent=L, topology=(Periodic, Periodic, Flat), halo=(3, 3))
            model = ShallowWaterModel(grid; gravitational_acceleration=1)

            x, y, z = nodes(model.grid, (Face(), Center(), Center()), reshape=true)

            uh₀(x, y) = x * y^2
            uh_answer = @. x * y^2

            h₀ = rand(size(grid)...)
            h_answer = deepcopy(h₀)

            set!(model, uh=uh₀, h=h₀)

            uh, vh, h = model.solution

            @test all(Array(interior(uh)) .≈ uh_answer)
            @test all(Array(interior(h)) .≈ h_answer)
        end
    end

    @testset "OctaHEALPix shallow-water vector-invariant tracer transport" begin
        @info "  Testing OctaHEALPix shallow-water vector-invariant tracer transport..."

        for FT in float_types
            shallow_water_vector_invariant_octahealpix_tracer_transport_uses_nonorthogonal_fluxes(FT, 4)
        end
    end

    @testset "OctaHEALPix shallow-water vector-invariant pressure gradient" begin
        @info "  Testing OctaHEALPix shallow-water vector-invariant pressure gradient..."

        for FT in float_types
            shallow_water_vector_invariant_octahealpix_pressure_gradient_uses_covariant_gradient(FT, 4)
        end
    end

    @testset "OctaHEALPix shallow-water conservative model support" begin
        @info "  Testing OctaHEALPix shallow-water conservative model support..."

        for FT in float_types
            shallow_water_conservative_octahealpix_model_update_uses_paired_compute(FT, 4)
            shallow_water_conservative_octahealpix_time_step_works(FT, 4)
            shallow_water_conservative_octahealpix_closure_field_refreshes_halos(FT, 4)
            shallow_water_conservative_octahealpix_tracer_tendency_uses_velocity_divergence(FT, 4)
        end
    end

    @testset "OctaHEALPix shallow-water cell advection timescale" begin
        @info "  Testing OctaHEALPix shallow-water cell advection timescale..."

        for FT in float_types
            shallow_water_octahealpix_cell_advection_timescale_uses_transport_scaling(FT, 4)
        end
    end

    @testset "OctaHEALPix shallow-water bounds-preserving tracer stepping" begin
        @info "  Testing OctaHEALPix shallow-water bounds-preserving tracer stepping..."

        for FT in float_types
            shallow_water_octahealpix_bounds_preserving_tracer_time_step_works(FT, 4)
        end
    end

    @testset "OctaHEALPix shallow-water mass tendencies" begin
        @info "  Testing OctaHEALPix shallow-water mass tendencies..."

        for FT in float_types
            shallow_water_octahealpix_mass_tendency_uses_nonorthogonal_transport(FT, 4)
        end
    end

    @testset "OctaHEALPix shallow-water vector-invariant momentum tendencies" begin
        @info "  Testing OctaHEALPix shallow-water vector-invariant momentum tendencies..."

        for FT in float_types
            shallow_water_octahealpix_vector_invariant_momentum_tendencies_match_live_advection(FT, 4)
        end
    end

    @testset "OctaHEALPix shallow-water AB2 vector-invariant momentum step" begin
        @info "  Testing OctaHEALPix shallow-water AB2 vector-invariant momentum step..."

        for FT in float_types
            shallow_water_octahealpix_ab2_step_updates_vector_invariant_momentum(FT, 4)
        end
    end

    @testset "OctaHEALPix shallow-water multistep end-to-end transport" begin
        @info "  Testing OctaHEALPix shallow-water multistep end-to-end transport..."

        for FT in float_types
            shallow_water_octahealpix_multistep_end_to_end_conserves_mass_and_tracer_mass(FT, 4)
        end
    end

    for arch in archs
        for topo in topos
            @testset "Time-stepping ShallowWaterModels [$arch, $topo]" begin
                @info "  Testing time-stepping ShallowWaterModels [$arch, $topo]..."
                @test time_stepping_shallow_water_model_works(arch, topo, nothing, nothing)
            end
        end

        for coriolis in (nothing, FPlane(f=1), BetaPlane(f₀=1, β=0.1))
            @testset "Time-stepping ShallowWaterModels [$arch, $(typeof(coriolis))]" begin
                @info "  Testing time-stepping ShallowWaterModels [$arch, $(typeof(coriolis))]..."
                @test time_stepping_shallow_water_model_works(arch, topos[1], coriolis, nothing)
            end
        end

        @testset "Time-step Wizard ShallowWaterModels [$arch, $topos[1]]" begin
        @info "  Testing time-step wizard ShallowWaterModels [$arch, $topos[1]]..."
            @test time_step_wizard_shallow_water_model_works(archs[1], topos[1], nothing)
        end

        # Advection = nothing is broken as halo does not have a maximum
        for advection in (nothing, Centered(), WENO())
            @testset "Time-stepping ShallowWaterModels [$arch, $(typeof(advection))]" begin
                @info "  Testing time-stepping ShallowWaterModels [$arch, $(typeof(advection))]..."
                @test time_stepping_shallow_water_model_works(arch, topos[1], nothing, advection)
            end
        end

        for timestepper in (:RungeKutta3, :QuasiAdamsBashforth2)
            @testset "Time-stepping ShallowWaterModels [$arch, $timestepper]" begin
                @info "  Testing time-stepping ShallowWaterModels [$arch, $timestepper]..."
                @test time_stepping_shallow_water_model_works(arch, topos[1], nothing, nothing, timestepper=timestepper)
            end
        end

        @testset "ShallowWaterModel with tracers and forcings [$arch]" begin
            @info "  Testing ShallowWaterModel with tracers and forcings [$arch]..."
            shallow_water_model_tracers_and_forcings_work(arch)
        end

        @testset "ShallowWaterModel viscous diffusion [$arch]" begin
            Nx, Ny = 10, 12
            grid_x = RectilinearGrid(arch, size = Nx, x = (0, 1), topology = (Bounded, Flat, Flat))
            grid_y = RectilinearGrid(arch, size = Ny, y = (0, 1), topology = (Flat, Bounded, Flat))
            coords = (reshape(xnodes(grid_x, Face()), (Nx+1, 1)), reshape(ynodes(grid_y, Face()), (1, Ny+1)))

            for (fieldname, grid, coord) in zip([:u, :v], [grid_x, grid_y], coords)
                for formulation in (ConservativeFormulation(), VectorInvariantFormulation())
                    @info "  Testing ShallowWaterModel cosine viscous diffusion [$fieldname, $formulation]"
                    test_shallow_water_diffusion_cosine(grid, formulation, fieldname, coord)
                end
            end
        end
    end

    @testset "ShallowWaterModels with ImmersedBoundaryGrid" begin
        for arch in archs
            @testset "ShallowWaterModels with ImmersedBoundaryGrid [$arch]" begin
                @info "Testing ShallowWaterModels with ImmersedBoundaryGrid [$arch]"

                # Gaussian bump of width "1"
                bump(x, y) = y < exp(-x^2)
                grid = RectilinearGrid(arch, size=(8, 8), x=(-10, 10), y=(0, 5), topology=(Periodic, Bounded, Flat))
                grid_with_bump = ImmersedBoundaryGrid(grid, GridFittedBoundary(bump))

                @test_throws ArgumentError model = ShallowWaterModel(grid_with_bump; gravitational_acceleration=1)

                grid = RectilinearGrid(arch, size=(8, 8), x=(-10, 10), y=(0, 5), topology=(Periodic, Bounded, Flat), halo=(4, 4))
                grid_with_bump = ImmersedBoundaryGrid(grid, GridFittedBoundary(bump))

                model = ShallowWaterModel(grid_with_bump; gravitational_acceleration=1)

                set!(model, h=1)
                simulation = Simulation(model, Δt=1.0, stop_iteration=1)
                run!(simulation)

                @test model.clock.iteration == 1
            end
        end
    end
end
