function run_thermal_bubble_regression_test(arch)
    Nx, Ny, Nz = 16, 16, 16
    Lx, Ly, Lz = 100, 100, 100
    Δt = 6

    grid = RegularCartesianGrid(size=(Nx, Ny, Nz), length=(Lx, Ly, Lz))
    closure = ConstantIsotropicDiffusivity(ν=4e-2, κ=4e-2)
    model = Model(architecture=arch, grid=grid, closure=closure, coriolis=FPlane(f=1e-4))

    model.tracers.T.data.parent .= 9.85
    model.tracers.S.data.parent .= 35.0

    # Add a cube-shaped warm temperature anomaly that takes up the middle 50%
    # of the domain volume.
    i1, i2 = round(Int, Nx/4), round(Int, 3Nx/4)
    j1, j2 = round(Int, Ny/4), round(Int, 3Ny/4)
    k1, k2 = round(Int, Nz/4), round(Int, 3Nz/4)
    model.tracers.T.data[i1:i2, j1:j2, k1:k2] .+= 0.01

    regression_data_filepath = joinpath(dirname(@__FILE__), "data", "thermal_bubble_regression.nc")

    ####
    #### Uncomment to generate regression data.
    ####

    #=
    @warn ("Generating new data for the thermal bubble regression test.")

    outputs = Dict("v" => model.velocities.v,
                   "u" => model.velocities.u,
                   "w" => model.velocities.w,
                   "T" => model.tracers.T,
                   "S" => model.tracers.S)

    nc_writer = NetCDFOutputWriter(model, outputs, filename=regression_data_filepath, frequency=10)
    push!(model.output_writers, nc_writer)
    =#

    ####
    #### Regression test
    ####

    time_step!(model, 10, Δt)

    ds = Dataset(regression_data_filepath, "r")

    u = ds["u"][:, :, :, end]
    v = ds["v"][:, :, :, end]
    w = ds["w"][:, :, :, end]
    T = ds["T"][:, :, :, end]
    S = ds["S"][:, :, :, end]

    field_names = ["u", "v", "w", "T", "S"]
    fields = [model.velocities.u, model.velocities.v, model.velocities.w, model.tracers.T, model.tracers.S]
    fields_correct = [u, v, w, T, S]
    summarize_regression_test(field_names, fields, fields_correct)

    # Now test that the model state matches the regression output.
    @test all(Array(interior(model.velocities.u)) .≈ u)
    @test all(Array(interior(model.velocities.v)) .≈ v)
    @test all(Array(interior(model.velocities.w)) .≈ w)
    @test all(Array(interior(model.tracers.T))    .≈ T)
    @test all(Array(interior(model.tracers.S))    .≈ S)
end
