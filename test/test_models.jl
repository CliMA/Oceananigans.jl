function set_velocity_tracer_fields(arch, grid, fieldname, value, answer)
    model = Model(architecture=arch, float_type=eltype(grid), grid=grid)
    kwarg = Dict(fieldname=>value)
    set!(model; kwarg...)

    if fieldname ∈ propertynames(model.velocities)
        ϕ = getproperty(model.velocities, fieldname)
    else
        ϕ = getproperty(model.tracers, fieldname)
    end

    return data(ϕ) ≈ answer
end

function initial_conditions_correctly_set(arch, FT)
    model = Model(grid=RegularCartesianGrid(FT; size=(16, 16, 8), length=(1, 2, 3)), architecture=arch, float_type=FT)

    # Set initial condition to some basic function we can easily check for.
    # We offset the functions by an integer so that we don't end up comparing
    # zero values with other zero values. I was too lazy to pick clever functions.
    u₀(x, y, z) = 1 + x+y+z
    v₀(x, y, z) = 2 + sin(x*y*z)
    w₀(x, y, z) = 3 + y*z
    T₀(x, y, z) = 4 + tanh(x+y-z)
    S₀(x, y, z) = 5

    set_ic!(model; u=u₀, v=v₀, w=w₀, T=T₀, S=S₀)

    Nx, Ny, Nz = model.grid.Nx, model.grid.Ny, model.grid.Nz
    xC, yC, zC = model.grid.xC, model.grid.yC, model.grid.zC
    xF, yF, zF = model.grid.xF, model.grid.yF, model.grid.zF
    u, v, w = model.velocities.u.data, model.velocities.v.data, model.velocities.w.data
    T, S = model.tracers.T.data, model.tracers.S.data

    all_values_match = true
    for i in 1:Nx, j in 1:Ny, k in 1:Nz
        values_match = ( u[i, j, k] ≈ 1 + xF[i] + yC[j] + zC[k]       &&
                         v[i, j, k] ≈ 2 + sin(xC[i] * yF[j] * zC[k])  &&
                         w[i, j, k] ≈ 3 + yC[j] * zF[k]               &&
                         T[i, j, k] ≈ 4 + tanh(xC[i] + yC[j] - zC[k]) &&
                         S[i, j, k] ≈ 5)
        all_values_match = all_values_match & values_match
    end

    return all_values_match
end

@testset "Models" begin
    println("Testing models...")

    @testset "Doubly periodic model" begin
        println("  Testing doubly periodic model construction...")
        for arch in archs, FT in float_types
            model = Model(grid=RegularCartesianGrid(FT; size=(16, 16, 2), length=(1, 2, 3)), architecture=arch, float_type=FT)

            # Just testing that a Model was constructed with no errors/crashes.
            @test true
        end
    end

    @testset "Reentrant channel model" begin
        println("  Testing reentrant channel model construction...")
        for arch in archs, FT in float_types
            model = ChannelModel(grid=RegularCartesianGrid(FT; size=(16, 16, 2), length=(1, 2, 3)), architecture=arch, float_type=FT)

            # Just testing that a ChannelModel was constructed with no errors/crashes.
            @test true
        end
    end

    @testset "Non-dimensional model" begin
        println("  Testing non-dimensional model construction...")
        for arch in archs, FT in float_types
            grid = RegularCartesianGrid(FT; size=(16, 16, 2), length=(3, 2, 1))
            model = NonDimensionalModel(architecture=arch, float_type=FT, grid=grid, Re=1, Pr=1, Ro=Inf)

            # Just testing that a NonDimensionalModel was constructed with no errors/crashes.
            @test true
        end
    end

    @testset "Setting model fields" begin
        println("  Testing setting model fields...")
        for arch in archs, FT in float_types
            N = (16, 16, 8)
            L = (2π, 3π, 5π)

            grid = RegularCartesianGrid(FT; size=N, length=L)
            xF = reshape(grid.xF[1:end-1], N[1], 1, 1)
            yC = reshape(grid.yC, 1, N[2], 1)
            zC = reshape(grid.zC, 1, 1, N[3])

            u₀(x, y, z) = x * y^2 * z^3
            u_answer = @. xF * yC^2 * zC^3

            T₀ = rand(size(grid)...)
            T_answer = deepcopy(T₀)

            @test set_velocity_tracer_fields(arch, grid, :u, u₀, u_answer)
            @test set_velocity_tracer_fields(arch, grid, :T, T₀, T_answer)

            @test initial_conditions_correctly_set(arch, FT)
        end
    end
end
