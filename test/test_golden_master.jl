using Random
const seed = 420  # Random seed to use for all pseudorandom number generators.

function run_thermal_bubble_golden_master_tests(arch)
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
                                   prefix="thermal_bubble_golden_master_",
                                   frequency=10, padding=2)

    # Uncomment to include a NetCDF output writer that produces the golden master.
    # push!(model.output_writers, nc_writer)

    time_step!(model, 10, Δt)

    u = read_output(nc_writer, "u", 10)
    v = read_output(nc_writer, "v", 10)
    w = read_output(nc_writer, "w", 10)
    T = read_output(nc_writer, "T", 10)
    S = read_output(nc_writer, "S", 10)

    # Now test that the model state matches the golden master output.
    @test all(u .≈ Array(model.velocities.u.data[1:Nx, 1:Ny, 1:Nz]))
    @test all(v .≈ Array(model.velocities.v.data[1:Nx, 1:Ny, 1:Nz]))
    @test all(w .≈ Array(model.velocities.w.data[1:Nx, 1:Ny, 1:Nz]))
    @test all(T .≈ Array(model.tracers.T.data[1:Nx, 1:Ny, 1:Nz]))
    @test all(S .≈ Array(model.tracers.S.data[1:Nx, 1:Ny, 1:Nz]))
end

function run_deep_convection_golden_master_tests()
    Nx, Ny, Nz = 32, 32, 16
    Lx, Ly, Lz = 2000, 2000, 1000
    Δt = 20

    model = Model(N=(Nx, Ny, Nz), L=(Lx, Ly, Lz), ν=4e-2, κ=4e-2)

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

    model.forcing = Forcing(nothing, nothing, nothing, cooling_disk, nothing)

    rng = MersenneTwister(seed)
    model.tracers.T.data[1:Nx, 1:Ny, 1] .+= 0.01*rand(rng, Nx, Ny)

    nc_writer = NetCDFOutputWriter(dir=".",
                                   prefix="deep_convection_golden_master_",
                                   frequency=10, padding=2)

    # Uncomment to include a NetCDF output writer that produces the golden master.
    # push!(model.output_writers, nc_writer)

    time_step!(model, 10, Δt)

    u = read_output(nc_writer, "u", 10)
    v = read_output(nc_writer, "v", 10)
    w = read_output(nc_writer, "w", 10)
    T = read_output(nc_writer, "T", 10)
    S = read_output(nc_writer, "S", 10)

    # Now test that the model state matches the golden master output.
    @test_skip all(u .≈ Array(model.velocities.u.data[1:Nx, 1:Ny, 1:Nz]))
    @test_skip all(v .≈ Array(model.velocities.v.data[1:Nx, 1:Ny, 1:Nz]))
    @test_skip all(w .≈ Array(model.velocities.w.data[1:Nx, 1:Ny, 1:Nz]))
    @test_skip all(T .≈ Array(model.tracers.T.data[1:Nx, 1:Ny, 1:Nz]))
    @test_skip all(S .≈ Array(model.tracers.S.data[1:Nx, 1:Ny, 1:Nz]))
end
