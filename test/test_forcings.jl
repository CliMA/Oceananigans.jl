add_one(args...) = 1.0

function test_forcing(fld)
    kwarg = Dict(Symbol(:F, fld) => add_one)
    forcing = Forcing(; kwarg...)
    f = getfield(forcing, fld)
    f() == 1.0
end

function time_step_with_forcing_function(arch)
    @inline Fu(grid, U, Φ, i, j, k) = @inbounds ifelse(k == grid.Nz, -u[i, j, k] / 60, 0)
    @inline Fv(grid, U, Φ, i, j, k) = @inbounds ifelse(k == grid.Nz, -v[i, j, k] / 60, 0)
    @inline Fw(grid, U, Φ, i, j, k) = @inbounds ifelse(k == grid.Nz, -w[i, j, k] / 60, 0)

    forcing = Forcing(Fu=Fu, Fv=Fv, Fw=Fw)

    model = Model(N=(8, 8, 8), L=(1, 1, 1), arch=arch, forcing=forcing)
    time_step!(model, 1, 1)
    return true
end

const τ = 60
function time_step_with_forcing_function_const(arch)
    @inline Fu(grid, U, Φ, i, j, k) = @inbounds ifelse(k == grid.Nz, -u[i, j, k] / τ, 0)
    @inline Fv(grid, U, Φ, i, j, k) = @inbounds ifelse(k == grid.Nz, -v[i, j, k] / τ, 0)
    @inline Fw(grid, U, Φ, i, j, k) = @inbounds ifelse(k == grid.Nz, -w[i, j, k] / τ, 0)

    forcing = Forcing(Fu=Fu, Fv=Fv, Fw=Fw)

    model = Model(N=(8, 8, 8), L=(1, 1, 1), arch=arch, forcing=forcing)
    time_step!(model, 1, 1)
    return true
end

@testset "Forcings" begin
    println("Testing forcings...")

    for fld in (:u, :v, :w, :T, :S)
        @test test_forcing(fld)
    end

    for arch in archs
        @test time_step_with_forcing_function(arch)
        @test time_step_with_forcing_function_const(arch)
    end
end
