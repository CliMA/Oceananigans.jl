function run_thermal_bubble_regression_test(arch, grid_type)
    Nx, Ny, Nz = 16, 16, 16
    Lx, Ly, Lz = 100, 100, 100
    Δt = 6

    if grid_type == :regular
        grid = RectilinearGrid(arch, size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz), halo=(1, 1, 1))
    elseif grid_type == :vertically_unstretched
        zF = range(-Lz, 0, length=Nz+1)
        grid = RectilinearGrid(arch, size=(Nx, Ny, Nz), x=(0, Lx), y=(0, Ly), z=zF, halo=(1, 1, 1))
    end

    closure = ScalarDiffusivity(ν=4e-2, κ=4e-2)
    model = NonhydrostaticModel(; grid, closure, coriolis=FPlane(f=1e-4),
                                buoyancy=SeawaterBuoyancy(), tracers=(:T, :S))
    simulation = Simulation(model, Δt=6, stop_iteration=10)

    model.tracers.T.data.parent .= 9.85
    model.tracers.S.data.parent .= 35.0

    # Add a cube-shaped warm temperature anomaly that takes up the middle 50%
    # of the domain volume.
    i1, i2 = round(Int, Nx/4), round(Int, 3Nx/4)
    j1, j2 = round(Int, Ny/4), round(Int, 3Ny/4)
    k1, k2 = round(Int, Nz/4), round(Int, 3Nz/4)
    view(model.tracers.T, i1:i2, j1:j2, k1:k2) .+= 0.01

    datadep_path = "regression_test_data/thermal_bubble_regression.nc"
    regression_data_filepath = @datadep_str datadep_path

    ####
    #### Uncomment the block below to generate regression data.
    ####

    #=
    @warn ("You are generating new data for the thermal bubble regression test.")

    outputs = Dict("v" => model.velocities.v,
                   "u" => model.velocities.u,
                   "w" => model.velocities.w,
                   "T" => model.tracers.T,
                   "S" => model.tracers.S)

    nc_writer = NetCDFOutputWriter(model, outputs, filename=regression_data_filepath, schedule=IterationInterval(10))
    push!(simulation.output_writers, nc_writer)
    =#

    ####
    #### Regression test
    ####

    run!(simulation)

    ds = Dataset(regression_data_filepath, "r")

    test_fields = (u = zeros(size(model.velocities.u)),
                   v = zeros(size(model.velocities.v)),
                   w = zeros(size(model.velocities.w)),
                   T = zeros(size(model.tracers.T)),
                   S = zeros(size(model.tracers.S)))

    copyto!(test_fields.u, interior(model.velocities.u))
    copyto!(test_fields.v, interior(model.velocities.v))
    copyto!(test_fields.w, interior(model.velocities.w))
    copyto!(test_fields.T, interior(model.tracers.T))
    copyto!(test_fields.S, interior(model.tracers.S))

    correct_fields = (u = ds["u"][:, :, :, end],
                      v = ds["v"][:, :, :, end],
                      w = ds["w"][:, :, :, end],
                      T = ds["T"][:, :, :, end],
                      S = ds["S"][:, :, :, end])

    summarize_regression_test(test_fields, correct_fields)
    
    @test all(test_fields.u .≈ correct_fields.u)
    @test all(test_fields.v .≈ correct_fields.v)
    @test all(test_fields.w .≈ correct_fields.w)
    @test all(test_fields.T .≈ correct_fields.T)
    @test all(test_fields.S .≈ correct_fields.S)
    
    return nothing
end
