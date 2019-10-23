add_one(args...) = 1.0

function test_forcing(fld)
    kwarg = Dict(fld => add_one)
    forcing = ModelForcing(; kwarg...)
    f = getfield(forcing, fld)
    return f() == 1.0
end

function time_step_with_forcing_functions(arch)
    @inline Fu(i, j, k, grid, time, U, Φ, params) = @inbounds ifelse(k == grid.Nz, -U.u[i, j, k] / 60, 0)
    @inline Fv(i, j, k, grid, time, U, Φ, params) = @inbounds ifelse(k == grid.Nz, -U.v[i, j, k] / 60, 0)
    @inline Fw(i, j, k, grid, time, U, Φ, params) = @inbounds ifelse(k == grid.Nz, -U.w[i, j, k] / 60, 0)

    forcing = ModelForcing(u=Fu, v=Fv, w=Fw)

    grid = RegularCartesianGrid(size=(16, 16, 16), length=(1, 1, 1))
    model = Model(grid=grid, architecture=arch, forcing=forcing)
    time_step!(model, 1, 1)
    return true
end

function time_step_with_forcing_functions_params(arch)
    @inline Fu(i, j, k, grid, time, U, Φ, params) = @inbounds ifelse(k == grid.Nz, -U.u[i, j, k] / params.τ, 0)
    @inline Fv(i, j, k, grid, time, U, Φ, params) = @inbounds ifelse(k == grid.Nz, -U.v[i, j, k] / params.τ, 0)
    @inline Fw(i, j, k, grid, time, U, Φ, params) = @inbounds ifelse(k == grid.Nz, -U.w[i, j, k] / params.τ, 0)

    forcing = ModelForcing(u=Fu, v=Fv, w=Fw)

    grid = RegularCartesianGrid(size=(16, 16, 16), length=(1, 1, 1))
    model = Model(grid=grid, architecture=arch, forcing=forcing, parameters=(τ=60,))
    time_step!(model, 1, 1)
    return true
end

function time_step_with_forcing_functions_sin_exp(arch)
    @inline Fu(i, j, k, grid, time, U, Φ, params) = @inbounds sin(grid.xC[i])
    @inline FT(i, j, k, grid, time, U, Φ, params) = @inbounds exp(-Φ.T[i, j, k])

    forcing = ModelForcing(u=Fu, T=FT)

    grid = RegularCartesianGrid(size=(16, 16, 16), length=(1, 1, 1))
    model = Model(grid=grid, architecture=arch, forcing=forcing)
    time_step!(model, 1, 1)
    return true
end

function time_step_with_simple_forcing(arch)
    u_forcing = SimpleForcing((x, y, z, t) -> sin(x))
    grid = RegularCartesianGrid(size=(16, 16, 16), length=(1, 1, 1))
    model = Model(grid=grid, architecture=arch, forcing=ModelForcing(u=u_forcing))
    time_step!(model, 1, 1)
    return true
end

function time_step_with_simple_forcing_parameters(arch)
    u_forcing = SimpleForcing((x, y, z, t, p) -> sin(p.ω * x), parameters=(ω=π,))
    grid = RegularCartesianGrid(size=(16, 16, 16), length=(1, 1, 1))
    model = Model(grid=grid, architecture=arch, forcing=ModelForcing(u=u_forcing))
    time_step!(model, 1, 1)
    return true
end

@testset "ModelForcing" begin
    println("Testing forcings...")

    @testset "Forcing function initialization" begin
        println("  Testing forcing function initialization...")
        for fld in (:u, :v, :w, :T, :S)
            @test test_forcing(fld)
        end
    end

    for arch in archs
        @testset "Forcing function time stepping [$(typeof(arch))]" begin
            println("  Testing forcing function time stepping [$(typeof(arch))]...")
            @test time_step_with_forcing_functions(arch)
            @test time_step_with_forcing_functions_params(arch)
            @test time_step_with_forcing_functions_sin_exp(arch)
            @test time_step_with_simple_forcing(arch)
            @test time_step_with_simple_forcing_parameters(arch)
        end
    end
end
