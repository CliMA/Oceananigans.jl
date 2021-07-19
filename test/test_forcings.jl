""" Take one time step with three forcing functions on u, v, w. """
function time_step_with_forcing_functions(arch)
    @inline Fu(x, y, z, t) = exp(π * z)
    @inline Fv(x, y, z, t) = cos(42 * x)
    @inline Fw(x, y, z, t) = 1.0

    grid = RegularRectilinearGrid(size=(1, 1, 1), extent=(1, 1, 1))
    model = NonhydrostaticModel(grid=grid, architecture=arch, forcing=(u=Fu, v=Fv, w=Fw))
    time_step!(model, 1, euler=true)

    return true
end

@inline Fu_discrete_func(i, j, k, grid, clock, model_fields) = @inbounds -model_fields.u[i, j, k]
@inline Fv_discrete_func(i, j, k, grid, clock, model_fields, params) = @inbounds - model_fields.v[i, j, k] / params.τ
@inline Fw_discrete_func(i, j, k, grid, clock, model_fields, params) = @inbounds - model_fields.w[i, j, k]^2 / params.τ

""" Take one time step with a DiscreteForcing function. """
function time_step_with_discrete_forcing(arch)

    Fu = Forcing(Fu_discrete_func, discrete_form=true)

    grid = RegularRectilinearGrid(size=(1, 1, 1), extent=(1, 1, 1))
    model = NonhydrostaticModel(grid=grid, architecture=arch, forcing=(u=Fu,))
    time_step!(model, 1, euler=true)

    return true
end

""" Take one time step with ParameterizedForcing forcing functions. """
function time_step_with_parameterized_discrete_forcing(arch)

    Fv = Forcing(Fv_discrete_func, parameters=(τ=60,), discrete_form=true)
    Fw = Forcing(Fw_discrete_func, parameters=(τ=60,), discrete_form=true)

    grid = RegularRectilinearGrid(size=(1, 1, 1), extent=(1, 1, 1))
    model = NonhydrostaticModel(grid=grid, architecture=arch, forcing=(v=Fv, w=Fw))
    time_step!(model, 1, euler=true)

    return true
end

""" Take one time step with a Forcing forcing function with parameters. """
function time_step_with_parameterized_continuous_forcing(arch)

    u_forcing = Forcing((x, y, z, t, ω) -> sin(ω * x), parameters=π)

    grid = RegularRectilinearGrid(size=(1, 1, 1), extent=(1, 1, 1))
    model = NonhydrostaticModel(grid=grid, architecture=arch, forcing=(u=u_forcing,))
    time_step!(model, 1, euler=true)

    return true
end

""" Take one time step with a Forcing forcing function with parameters. """
function time_step_with_single_field_dependent_forcing(arch, fld)

    forcing = NamedTuple{(fld,)}((Forcing((x, y, z, t, u) -> -u, field_dependencies=:u),))

    grid = RegularRectilinearGrid(size=(1, 1, 1), extent=(1, 1, 1))
    model = NonhydrostaticModel(grid=grid, architecture=arch, forcing=forcing)
    time_step!(model, 1, euler=true)

    return true
end

""" Take one time step with a Forcing forcing function with parameters. """
function time_step_with_multiple_field_dependent_forcing(arch)

    u_forcing = Forcing((x, y, z, t, v, w, T) -> sin(v) * exp(w) * T, field_dependencies=(:v, :w, :T))

    grid = RegularRectilinearGrid(size=(1, 1, 1), extent=(1, 1, 1))
    model = NonhydrostaticModel(grid=grid, architecture=arch, forcing=(u=u_forcing,))
    time_step!(model, 1, euler=true)

    return true
end



""" Take one time step with a Forcing forcing function with parameters. """
function time_step_with_parameterized_field_dependent_forcing(arch)

    u_forcing = Forcing((x, y, z, t, u, p) -> sin(p.ω * x) * u, parameters=(ω=π,), field_dependencies=:u)

    grid = RegularRectilinearGrid(size=(1, 1, 1), extent=(1, 1, 1))
    model = NonhydrostaticModel(grid=grid, architecture=arch, forcing=(u=u_forcing,))
    time_step!(model, 1, euler=true)

    return true
end

function relaxed_time_stepping(arch)
    x_relax = Relaxation(rate = 1/60,   mask = GaussianMask{:x}(center=0.5, width=0.1), 
                                      target = LinearTarget{:x}(intercept=π, gradient=ℯ))

    y_relax = Relaxation(rate = 1/60,   mask = GaussianMask{:y}(center=0.5, width=0.1),
                                      target = LinearTarget{:y}(intercept=π, gradient=ℯ))

    z_relax = Relaxation(rate = 1/60,   mask = GaussianMask{:z}(center=0.5, width=0.1),
                                      target = π)

    grid = RegularRectilinearGrid(size=(1, 1, 1), extent=(1, 1, 1))
    model = NonhydrostaticModel(grid=grid, architecture=arch, forcing=(u=x_relax, v=y_relax, w=z_relax))
    time_step!(model, 1, euler=true)

    return true
end

@testset "Forcings" begin
    @info "Testing forcings..."

    for arch in archs
        @testset "Forcing function time stepping [$(typeof(arch))]" begin
            @info "  Testing forcing function time stepping [$(typeof(arch))]..."

            @testset "Non-parameterized forcing functions [$(typeof(arch))]" begin
                @info "      Testing non-parameterized forcing functions [$(typeof(arch))]..."
                @test time_step_with_forcing_functions(arch)
                @test time_step_with_discrete_forcing(arch)
            end

            @testset "Parameterized forcing functions [$(typeof(arch))]" begin
                @info "      Testing parameterized forcing functions [$(typeof(arch))]..."
                @test time_step_with_parameterized_continuous_forcing(arch)
                @test time_step_with_parameterized_discrete_forcing(arch)
            end

            @testset "Field-dependent forcing functions [$(typeof(arch))]" begin
                @info "      Testing field-dependent forcing functions [$(typeof(arch))]..."

                for fld in (:u, :v, :w, :T)
                    @test time_step_with_single_field_dependent_forcing(arch, fld)
                end

                @test time_step_with_multiple_field_dependent_forcing(arch)
                @test time_step_with_parameterized_field_dependent_forcing(arch)
            end 

            @testset "Relaxation forcing functions [$(typeof(arch))]" begin
                @info "      Testing relaxation forcing functions [$(typeof(arch))]..."
                @test relaxed_time_stepping(arch)
            end
        end
    end
end
