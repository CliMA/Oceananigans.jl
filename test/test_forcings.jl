add_one(args...) = 1.0

""" Instantiate ModelForcing. """
function test_forcing(fld)
    kwarg = Dict(fld => add_one)
    forcing = ModelForcing(; kwarg...)
    f = getfield(forcing, fld)
    return f() == 1.0
end

""" Take one time step with three forcing functions on u, v, w. """
function time_step_with_forcing_functions(arch)
    @inline Fu(i, j, k, grid, clock, state) = @inbounds ifelse(k == grid.Nz, -state.velocities.u[i, j, k] / 60, 0)
    @inline Fv(i, j, k, grid, clock, state) = @inbounds ifelse(k == grid.Nz, -state.velocities.v[i, j, k] / 60, 0)
    @inline Fw(i, j, k, grid, clock, state) = @inbounds ifelse(k == grid.Nz, -state.velocities.w[i, j, k] / 60, 0)

    forcing = ModelForcing(u=Fu, v=Fv, w=Fw)

    grid = RegularCartesianGrid(size=(16, 16, 16), extent=(1, 1, 1))
    model = IncompressibleModel(grid=grid, architecture=arch, forcing=forcing)
    time_step!(model, 1, euler=true)

    return true
end

""" Take one time step with ParameterizedForcing forcing functions. """
function time_step_with_parameterized_forcing(arch)
    @inline Fu_func(i, j, k, grid, clock, state, params) = @inbounds ifelse(k == grid.Nz, -state.velocities.u[i, j, k] / params.τ, 0)
    @inline Fv_func(i, j, k, grid, clock, state, params) = @inbounds ifelse(k == grid.Nz, -state.velocities.v[i, j, k] / params.τ, 0)
    @inline Fw_func(i, j, k, grid, clock, state, params) = @inbounds ifelse(k == grid.Nz, -state.velocities.w[i, j, k] / params.τ, 0)

    Fu = ParameterizedForcing(Fu_func, (τ=60,))
    Fv = ParameterizedForcing(Fv_func, (τ=60,))
    Fw = ParameterizedForcing(Fw_func, (τ=60,))

    forcing = ModelForcing(u=Fu, v=Fv, w=Fw)

    grid = RegularCartesianGrid(size=(16, 16, 16), extent=(1, 1, 1))
    model = IncompressibleModel(grid=grid, architecture=arch, forcing=forcing)
    time_step!(model, 1, euler=true)

    return true
end

""" Take one time step with forcing functions containing sin and exp functions. """
function time_step_with_forcing_functions_sin_exp(arch)
    @inline Fu(i, j, k, grid, clock, state) = @inbounds sin(grid.xC[i])
    @inline FT(i, j, k, grid, clock, state) = @inbounds exp(-state.tracers.T[i, j, k])

    forcing = ModelForcing(u=Fu, T=FT)

    grid = RegularCartesianGrid(size=(16, 16, 16), extent=(1, 1, 1))
    model = IncompressibleModel(grid=grid, architecture=arch, forcing=forcing)
    time_step!(model, 1, euler=true)

    return true
end

""" Take one time step with a SimpleForcing forcing function. """
function time_step_with_simple_forcing(arch)
    u_forcing = SimpleForcing((x, y, z, t) -> sin(x))
    grid = RegularCartesianGrid(size=(16, 16, 16), extent=(1, 1, 1))
    model = IncompressibleModel(grid=grid, architecture=arch, forcing=ModelForcing(u=u_forcing))
    time_step!(model, 1, euler=true)
    return true
end

""" Take one time step with a SimpleForcing forcing function with parameters. """
function time_step_with_simple_forcing_parameters(arch)
    u_forcing = SimpleForcing((x, y, z, t, p) -> sin(p.ω * x), parameters=(ω=π,))
    grid = RegularCartesianGrid(size=(16, 16, 16), extent=(1, 1, 1))
    model = IncompressibleModel(grid=grid, architecture=arch, forcing=ModelForcing(u=u_forcing))
    time_step!(model, 1, euler=true)
    return true
end

""" Take one time step with a SimpleForcing forcing function with parameters. """
function time_step_with_simple_multiplicative_forcing(arch)
    u_forcing = SimpleForcing((x, y, z, t, u) -> -u, multiplicative=true)
    grid = RegularCartesianGrid(size=(16, 16, 16), extent=(1, 1, 1))
    model = IncompressibleModel(grid=grid, architecture=arch, forcing=ModelForcing(u=u_forcing))
    time_step!(model, 1, euler=true)
    return true
end

""" Take one time step with a SimpleForcing forcing function with parameters. """
function time_step_with_simple_multiplicative_forcing_parameters(arch)
    u_forcing = SimpleForcing((x, y, z, t, u, p) -> sin(p.ω * x) * u, parameters=(ω=π,), multiplicative=true)
    grid = RegularCartesianGrid(size=(16, 16, 16), extent=(1, 1, 1))
    model = IncompressibleModel(grid=grid, architecture=arch, forcing=ModelForcing(u=u_forcing))
    time_step!(model, 1, euler=true)
    return true
end

@testset "Forcing" begin
    @info "Testing forcings..."

    @testset "Forcing function initialization" begin
        @info "  Testing forcing function initialization..."
        for fld in (:u, :v, :w, :T, :S)
            @test test_forcing(fld)
        end
    end

    for arch in archs
        @testset "Forcing function time stepping [$(typeof(arch))]" begin
            @info "  Testing forcing function time stepping [$(typeof(arch))]..."
            @test time_step_with_forcing_functions(arch)
            @test time_step_with_parameterized_forcing(arch)
            @test time_step_with_forcing_functions_sin_exp(arch)
            @test time_step_with_simple_forcing(arch)
            @test time_step_with_simple_forcing_parameters(arch)
            @test time_step_with_simple_multiplicative_forcing(arch)
            @test time_step_with_simple_multiplicative_forcing_parameters(arch)
        end
    end
end
