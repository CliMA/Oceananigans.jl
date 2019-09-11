add_one(args...) = 1.0

function test_forcing(fld)
    kwarg = Dict(Symbol(:F, fld) => add_one)
    forcing = Forcing(; kwarg...)
    f = getfield(forcing, fld)
    f() == 1.0
end

function time_step_with_forcing_functions(arch)
    @inline Fu(i, j, k, grid, time, U, Φ, params) = @inbounds ifelse(k == grid.Nz, -U.u[i, j, k] / 60, 0)
    @inline Fv(i, j, k, grid, time, U, Φ, params) = @inbounds ifelse(k == grid.Nz, -U.v[i, j, k] / 60, 0)
    @inline Fw(i, j, k, grid, time, U, Φ, params) = @inbounds ifelse(k == grid.Nz, -U.w[i, j, k] / 60, 0)

    forcing = Forcing(Fu=Fu, Fv=Fv, Fw=Fw)

    model = Model(N=(16, 16, 16), L=(1, 1, 1), arch=arch, forcing=forcing)
    time_step!(model, 1, 1)
    return true
end

function time_step_with_forcing_functions_params(arch)
    @inline Fu(i, j, k, grid, time, U, Φ, params) = @inbounds ifelse(k == grid.Nz, -U.u[i, j, k] / params.τ, 0)
    @inline Fv(i, j, k, grid, time, U, Φ, params) = @inbounds ifelse(k == grid.Nz, -U.v[i, j, k] / params.τ, 0)
    @inline Fw(i, j, k, grid, time, U, Φ, params) = @inbounds ifelse(k == grid.Nz, -U.w[i, j, k] / params.τ, 0)

    forcing = Forcing(Fu=Fu, Fv=Fv, Fw=Fw)

    model = Model(N=(16, 16, 16), L=(1, 1, 1), arch=arch, forcing=forcing, parameters=(τ=60,))
    time_step!(model, 1, 1)
    return true
end


function time_step_with_forcing_functions_sin_exp(arch)
    @inline Fu(i, j, k, grid, time, U, Φ, params) = @inbounds sin(grid.xC[i])
    @inline FT(i, j, k, grid, time, U, Φ, params) = @inbounds exp(-Φ.T[i, j, k])

    forcing = Forcing(Fu=Fu, FT=FT)

    model = Model(N=(16, 16, 16), L=(1, 1, 1), arch=arch, forcing=forcing)
    time_step!(model, 1, 1)
    return true
end

@testset "Forcings" begin
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
        end
    end
end
