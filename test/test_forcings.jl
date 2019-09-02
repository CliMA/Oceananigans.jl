add_one(args...) = 1.0

function test_forcing(fld)
    kwarg = Dict(Symbol(:F, fld) => add_one)
    forcing = Forcing(; kwarg...)
    f = getfield(forcing, fld)
    f() == 1.0
end

function time_step_with_forcing_functions(arch)
    @inline Fu(grid, U, Φ, i, j, k) = @inbounds ifelse(k == grid.Nz, -U.u[i, j, k] / 60, 0)
    @inline Fv(grid, U, Φ, i, j, k) = @inbounds ifelse(k == grid.Nz, -U.v[i, j, k] / 60, 0)
    @inline Fw(grid, U, Φ, i, j, k) = @inbounds ifelse(k == grid.Nz, -U.w[i, j, k] / 60, 0)

    forcing = Forcing(Fu=Fu, Fv=Fv, Fw=Fw)

    model = Model(N=(16, 16, 16), L=(1, 1, 1), arch=arch, forcing=forcing)
    time_step!(model, 1, 1)
    return true
end

const τ = 60
function time_step_with_forcing_functions_const(arch)
    @inline Fu(grid, U, Φ, i, j, k) = @inbounds ifelse(k == grid.Nz, -U.u[i, j, k] / τ, 0)
    @inline Fv(grid, U, Φ, i, j, k) = @inbounds ifelse(k == grid.Nz, -U.v[i, j, k] / τ, 0)
    @inline Fw(grid, U, Φ, i, j, k) = @inbounds ifelse(k == grid.Nz, -U.w[i, j, k] / τ, 0)

    forcing = Forcing(Fu=Fu, Fv=Fv, Fw=Fw)

    model = Model(N=(16, 16, 16), L=(1, 1, 1), arch=arch, forcing=forcing)
    time_step!(model, 1, 1)
    return true
end


const α = 1
const β = 2
function time_step_with_forcing_functions_sin_exp(arch)
    @inline Fu(grid, U, Φ, i, j, k) = @inbounds sin(α * grid.xC[i])
    @inline FT(grid, U, Φ, i, j, k) = @inbounds exp(-β * Φ.T[i, j, k])

    forcing = Forcing(Fu=Fu, FT=FT)

    model = Model(N=(16, 16, 16), L=(1, 1, 1), arch=arch, forcing=forcing)
    time_step!(model, 1, 1)
    return true
end

@testset "Forcings" begin
    println("Testing forcings...")

    @testset "Forcing function initialization" begin
        println("Testing forcing function initialization...")
        for fld in (:u, :v, :w, :T, :S)
            @test test_forcing(fld)
        end
    end

    for arch in archs
        @testset "Forcing function time stepping [$(typeof(arch))]" begin
            println("Testing forcing function time stepping [$(typeof(arch))]...")
            @test time_step_with_forcing_functions(arch)
            @test time_step_with_forcing_functions_const(arch)
            @test time_step_with_forcing_functions_sin_exp(arch)
        end
    end
end
