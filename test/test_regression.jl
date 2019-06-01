using Random, Printf, JLD
const seed = 420  # Random seed to use for all pseudorandom number generators.

mutable struct JLDOutputWriter{N} <: OutputWriter
                 dir :: String
              prefix :: String
           fieldsets :: NTuple{N, Symbol}
    output_frequency :: Int
             padding :: Int
end

ext(::JLDOutputWriter) = ".jld"

function JLDOutputWriter(; dir=".", prefix="", fieldsets=(:velocities, :tracers, :G), frequency=1, padding=9)
    return JLDOutputWriter(dir, prefix, fieldsets, frequency, padding)
end

filename(iter, fw::JLDOutputWriter) = joinpath(fw.dir, fw.prefix * lpad(iter, fw.padding, "0") * ext(fw))

function Oceananigans.write_output(model, fw::JLDOutputWriter)
    filepath = filename(model.clock.iteration, fw)
    write_jld_output(Tuple(getproperty(model, s) for s in fw.fieldsets), filepath)
end

function Oceananigans.read_output(name, iter, fw::JLDOutputWriter)
    filepath = filename(iter, fw)
    data = load(filepath, name)
    return data
end

function write_jld_output(sets, filepath)
    allvars = Dict{String, Any}()
    for s in sets
        svars = Dict((String(fld), Array(getproperty(s, fld).data)) for fld in propertynames(s))
        merge!(allvars, svars)
    end

    save(filepath, allvars)

    return nothing
end

function run_thermal_bubble_regression_tests(arch)
    Nx, Ny, Nz = 16, 16, 16
    Lx, Ly, Lz = 100, 100, 100
    Δt = 6

    model = Model(N=(Nx, Ny, Nz), L=(Lx, Ly, Lz), arch=arch, ν=4e-2, κ=4e-2)

    # Add a cube-shaped warm temperature anomaly that takes up the middle 50%
    # of the domain volume.
    i1, i2 = round(Int, Nx/4), round(Int, 3Nx/4)
    j1, j2 = round(Int, Ny/4), round(Int, 3Ny/4)
    k1, k2 = round(Int, Nz/4), round(Int, 3Nz/4)
    model.tracers.T.data[i1:i2, j1:j2, k1:k2] .+= 0.01

    nc_writer = NetCDFOutputWriter(dir=".",
                                   prefix="thermal_bubble_regression_",
                                   frequency=10, padding=2)

    # Uncomment to include a NetCDF output writer that produces the regression.
    # push!(model.output_writers, nc_writer)

    time_step!(model, 10, Δt)

    u = read_output(nc_writer, "u", 10)
    v = read_output(nc_writer, "v", 10)
    w = read_output(nc_writer, "w", 10)
    T = read_output(nc_writer, "T", 10)
    S = read_output(nc_writer, "S", 10)

    field_names = ["u", "v", "w", "T", "S"]
    fields = [model.velocities.u, model.velocities.v, model.velocities.w, model.tracers.T, model.tracers.S]
    fields_gm = [u, v, w, T, S]
    for (field_name, φ, φ_gm) in zip(field_names, fields, fields_gm)
        φ_min = minimum(Array(data(φ)) - φ_gm)
        φ_max = maximum(Array(data(φ)) - φ_gm)
        φ_mean = mean(Array(data(φ)) - φ_gm)
        φ_abs_mean = mean(abs.(Array(data(φ)) - φ_gm))
        φ_std = std(Array(data(φ)) - φ_gm)
        @info(@sprintf("Δ%s: min=%.6g, max=%.6g, mean=%.6g, absmean=%.6g, std=%.6g\n",
                       field_name, φ_min, φ_max, φ_mean, φ_abs_mean, φ_std))
    end

    # Now test that the model state matches the regression output.
    @test all(Array(data(model.velocities.u)) .≈ u)
    @test all(Array(data(model.velocities.v)) .≈ v)
    @test all(Array(data(model.velocities.w)) .≈ w)
    @test all(Array(data(model.tracers.T))    .≈ T)
    @test all(Array(data(model.tracers.S))    .≈ S)
end

function run_deep_convection_regression_tests()
    Nx, Ny, Nz = 32, 32, 16
    Lx, Ly, Lz = 2000, 2000, 1000
    Δt = 20

    function cooling_disk(grid, u, v, w, T, S, i, j, k)
        if k == 1
            x = i*grid.Δx
            y = j*grid.Δy
            r² = (x - grid.Lx/2)^2 + (y - grid.Ly/2)^2
            if r² < 600^2
                return -4.5e-6
            else
                return 0
            end
        else
            return 0
        end
    end

    model = Model(N=(Nx, Ny, Nz), L=(Lx, Ly, Lz), ν=4e-2, κ=4e-2,
                  forcing=Forcing(nothing, nothing, nothing, cooling_disk, nothing))

    rng = MersenneTwister(seed)
    model.tracers.T.data[1:Nx, 1:Ny, 1] .+= 0.01*rand(rng, Nx, Ny)

    nc_writer = NetCDFOutputWriter(dir=".",
                                   prefix="deep_convection_regression_",
                                   frequency=10, padding=2)

    # Uncomment to include a NetCDF output writer that produces the regression.
    # push!(model.output_writers, nc_writer)

    time_step!(model, 10, Δt)

    u = read_output(nc_writer, "u", 10)
    v = read_output(nc_writer, "v", 10)
    w = read_output(nc_writer, "w", 10)
    T = read_output(nc_writer, "T", 10)
    S = read_output(nc_writer, "S", 10)

    field_names = ["u", "v", "w", "T", "S"]
    fields = [model.velocities.u, model.velocities.v, model.velocities.w, model.tracers.T, model.tracers.S]
    fields_gm = [u, v, w, T, S]
    for (field_name, φ, φ_gm) in zip(field_names, fields, fields_gm)
        φ_min = minimum(Array(data(φ)) - φ_gm)
        φ_max = maximum(Array(data(φ)) - φ_gm)
        φ_mean = mean(Array(data(φ)) - φ_gm)
        φ_abs_mean = mean(abs.(Array(data(φ)) - φ_gm))
        φ_std = std(Array(data(φ)) - φ_gm)
        @info(@sprintf("Δ%s: min=%.6g, max=%.6g, mean=%.6g, absmean=%.6g, std=%.6g\n",
                       field_name, φ_min, φ_max, φ_mean, φ_abs_mean, φ_std))
    end

    # Now test that the model state matches the regression output.
    @test_skip all(Array(data(model.velocities.u)) .≈ u)
    @test_skip all(Array(data(model.velocities.v)) .≈ v)
    @test_skip all(Array(data(model.velocities.w)) .≈ w)
    @test_skip all(Array(data(model.tracers.T))    .≈ T)
    @test_skip all(Array(data(model.tracers.S))    .≈ S)
end

function run_rayleigh_benard_regression_test(arch)

    #
    # Parameters
    #
          α = 2                 # aspect ratio
          n = 1                 # resolution multiple
         Ra = 1e6               # Rayleigh number
    Nx = Ny = 8n * α            # horizontal resolution
    Lx = Ly = 1.0 * α           # horizontal extent
         Nz = 16n               # vertical resolution
         Lz = 1.0               # vertical extent
         Pr = 0.7               # Prandtl number
          a = 1e-1              # noise amplitude for initial condition
         Δb = 1.0               # buoyancy differential

    # Rayleigh and Prandtl determine transport coefficients
    ν = sqrt(Δb * Pr * Lz^3 / Ra)
    κ = ν / Pr

    #
    # Model setup
    #

    # Force salinity as a passive tracer (βS=0)
    S★(x, z) = exp(4z) * sin(2π/Lx * x)
    FS(grid, u, v, w, T, S, i, j, k) = 1/10 * (S★(grid.xC[i], grid.zC[k]) - S[i, j, k])

    model = Model(
         arch = arch,
            N = (Nx, Ny, Nz),
            L = (Lx, Ly, Lz),
            ν = ν,
            κ = κ,
          eos = LinearEquationOfState(βT=1., βS=0.),
    constants = PlanetaryConstants(g=1., f=0.),
          bcs = BoundaryConditions(T=FieldBoundaryConditions(z=ZBoundaryConditions(
                       top = BoundaryCondition(Value, 0.0),
                    bottom = BoundaryCondition(Value, Δb)
                ))),
      forcing = Forcing(FS=FS)
    )

    ArrayType = typeof(model.velocities.u.data.parent)  # The type of the underlying data, not the offset array.
    Δt = 0.01 * min(model.grid.Δx, model.grid.Δy, model.grid.Δz)^2 / ν

    spinup_steps = 1000
      test_steps = 100

    prefix = "data_rayleigh_benard_regression_"
    outputwriter = JLDOutputWriter(dir=".", prefix=prefix, frequency=test_steps)

    #
    # Initial condition and spinup steps for creating regression test data
    #

    #=
    ξ(z) = a * rand() * z * (Lz + z) # noise, damped at the walls
    b₀(x, y, z) = (ξ(z) - z) / Lz

    x, y, z = model.grid.xC, model.grid.yC, model.grid.zC
    x, y, z = reshape(x, Nx, 1, 1), reshape(y, 1, Ny, 1), reshape(z, 1, 1, Nz)

    model.tracers.T.data .= ArrayType(b₀.(x, y, z))

    println("Spinning up... ")

    @time begin
        time_step!(model, spinup_steps-test_steps, Δt)
        push!(model.output_writers, outputwriter)
    end

    time_step!(model, 2test_steps, Δt)
    =#

    #
    # Regression test
    #

    # Load initial state
    u₀ = read_output("u",  spinup_steps, outputwriter)
    v₀ = read_output("v",  spinup_steps, outputwriter)
    w₀ = read_output("w",  spinup_steps, outputwriter)
    T₀ = read_output("T",  spinup_steps, outputwriter)
    S₀ = read_output("S",  spinup_steps, outputwriter)

    Gu = read_output("Gu", spinup_steps, outputwriter)
    Gv = read_output("Gv", spinup_steps, outputwriter)
    Gw = read_output("Gw", spinup_steps, outputwriter)
    GT = read_output("GT", spinup_steps, outputwriter)
    GS = read_output("GS", spinup_steps, outputwriter)

    data(model.velocities.u) .= ArrayType(u₀)
    data(model.velocities.v) .= ArrayType(v₀)
    data(model.velocities.w) .= ArrayType(w₀)
    data(model.tracers.T)    .= ArrayType(T₀)
    data(model.tracers.S)    .= ArrayType(S₀)

    data(model.G.Gu) .= ArrayType(Gu)
    data(model.G.Gv) .= ArrayType(Gv)
    data(model.G.Gw) .= ArrayType(Gw)
    data(model.G.GT) .= ArrayType(GT)
    data(model.G.GS) .= ArrayType(GS)

    model.clock.iteration = spinup_steps
    model.clock.time = spinup_steps * Δt

    # Step the model forward and perform the regression test
    time_step!(model, test_steps, Δt)

    u₁ = read_output("u", spinup_steps + test_steps, outputwriter)
    v₁ = read_output("v", spinup_steps + test_steps, outputwriter)
    w₁ = read_output("w", spinup_steps + test_steps, outputwriter)
    T₁ = read_output("T", spinup_steps + test_steps, outputwriter)
    S₁ = read_output("S", spinup_steps + test_steps, outputwriter)

    field_names = ["u", "v", "w", "T", "S"]
    fields = [model.velocities.u, model.velocities.v, model.velocities.w, model.tracers.T, model.tracers.S]
    fields_gm = [u₁, v₁, w₁, T₁, S₁]
    for (field_name, φ, φ_gm) in zip(field_names, fields, fields_gm)
        φ_min = minimum(Array(data(φ)) - φ_gm)
        φ_max = maximum(Array(data(φ)) - φ_gm)
        φ_mean = mean(Array(data(φ)) - φ_gm)
        φ_abs_mean = mean(abs.(Array(data(φ)) - φ_gm))
        φ_std = std(Array(data(φ)) - φ_gm)
        @info(@sprintf("Δ%s: min=%.6g, max=%.6g, mean=%.6g, absmean=%.6g, std=%.6g\n",
                       field_name, φ_min, φ_max, φ_mean, φ_abs_mean, φ_std))
    end

    # Now test that the model state matches the regression output.
    @test all(Array(data(model.velocities.u)) .≈ u₁)
    @test all(Array(data(model.velocities.v)) .≈ v₁)
    @test all(Array(data(model.velocities.w)) .≈ w₁)
    @test all(Array(data(model.tracers.T))    .≈ T₁)
    @test all(Array(data(model.tracers.S))    .≈ S₁)
    return nothing
end
