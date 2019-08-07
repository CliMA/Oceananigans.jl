function initial_conditions_correctly_set(arch, FT)
    model = Model(N=(16, 16, 16), L=(1, 2, 3), arch=arch, float_type=FT)

    # Set initial condition to some basic function we can easily check for.
    # We offset the functions by an integer so that we don't end up comparing
    # zero values with other zero values. I was too lazy to pick clever functions.
    u₀(x, y, z) = 1 + x+y+z
    v₀(x, y, z) = 2 + sin(x*y*z)
    w₀(x, y, z) = 3 + exp(y/z)
    T₀(x, y, z) = 4 + tanh(x+y-z)

    set_ic!(model; u=u₀, v=v₀, w=w₀, T=T₀)

    Nx, Ny, Nz = model.grid.Nx, model.grid.Ny, model.grid.Nz
    xC, yC, zC = model.grid.xC, model.grid.yC, model.grid.zC
    xF, yF, zF = model.grid.xF, model.grid.yF, model.grid.zF
    u, v, w = model.velocities.u.data, model.velocities.v.data, model.velocities.w.data
    T = model.tracers.T.data

    all_values_match = true
    for i in 1:Nx, j in 1:Ny, k in 1:Nz
        values_match = ( u[i, j, k] ≈ 1 + xF[i] + yC[j] + zC[k]      &&
                         v[i, j, k] ≈ 2 + sin(xC[i] * yF[j] * zC[k]) &&
                         w[i, j, k] ≈ 3 + exp(yC[j] / zF[k])         &&
                         T[i, j, k] ≈ 4 + tanh(xC[i] + yC[j] - zC[k]))
        all_values_match = all_values_match & values_match
    end

    return all_values_match
end

@testset "Models" begin
    println("Testing models...")

    @testset "Doubly periodic model" begin
        println("  Testing doubly periodic model construction...")
        for arch in archs, FT in float_types
            model = Model(N=(4, 5, 6), L=(1, 2, 3), arch=arch, float_type=FT)

            # Just testing that a Model was constructed with no errors/crashes.
            @test true
        end
    end

    @testset "Reentrant channel model" begin
        println("  Testing reentrant channel model construction...")
        for arch in archs, FT in float_types
            model = ChannelModel(N=(6, 5, 4), L=(3, 2, 1), arch=arch, float_type=FT)

            # Just testing that a ChannelModel was constructed with no errors/crashes.
            @test true
        end
    end

    @testset "Setting initial conditions" begin
        println("  Testing setting initial conditions...")
        for arch in archs, FT in float_types
            @test initial_conditions_correctly_set(arch, FT)
        end
    end
end
