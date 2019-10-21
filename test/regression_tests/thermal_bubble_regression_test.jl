
function run_thermal_bubble_regression_test(arch)
    Nx, Ny, Nz = 16, 16, 16
    Lx, Ly, Lz = 100, 100, 100
    Δt = 6

    model = BasicModel(N=(Nx, Ny, Nz), L=(Lx, Ly, Lz), architecture=arch, ν=4e-2, κ=4e-2,
                       coriolis=FPlane(f=1e-4))

    model.tracers.T.data.parent .= 9.85
    model.tracers.S.data.parent .= 35.0

    # Add a cube-shaped warm temperature anomaly that takes up the middle 50%
    # of the domain volume.
    i1, i2 = round(Int, Nx/4), round(Int, 3Nx/4)
    j1, j2 = round(Int, Ny/4), round(Int, 3Ny/4)
    k1, k2 = round(Int, Nz/4), round(Int, 3Nz/4)
    model.tracers.T.data[i1:i2, j1:j2, k1:k2] .+= 0.01

    outputs = Dict("v"=>model.velocities.v,
                   "u"=>model.velocities.u,
                   "w"=>model.velocities.w,
                   "T"=>model.tracers.T,
                   "S"=>model.tracers.S)
    nc_writer = NetCDFOutputWriter(model, outputs,
                                   filename="thermal_bubble_regression_test.nc",
                                   frequency=10)
    push!(model.output_writers, nc_writer)

    time_step!(model, 10, Δt)

    close(nc_writer)

    u = read_output(nc_writer, "u")
    v = read_output(nc_writer, "v")
    w = read_output(nc_writer, "w")
    T = read_output(nc_writer, "T")
    S = read_output(nc_writer, "S")

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


