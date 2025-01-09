include("dependencies_for_runtests.jl")

using TimesDates: TimeDate
using Dates: DateTime, Nanosecond, Millisecond
using TimesDates: TimeDate

using CUDA
using NCDatasets

using Oceananigans: Clock
using Oceananigans.Models.HydrostaticFreeSurfaceModels: VectorInvariant

#####
##### NetCDFOutputWriter tests
#####

function test_datetime_netcdf_output(arch)
    grid = RectilinearGrid(arch, size=(1, 1, 1), extent=(1, 1, 1))

    clock = Clock(time=DateTime(2021, 1, 1))

    model = NonhydrostaticModel(; grid, clock,
        timestepper = :QuasiAdamsBashforth2,
        buoyancy = SeawaterBuoyancy(),
        tracers=(:T, :S)
    )

    Δt = 5days + 3hours + 44.123seconds
    simulation = Simulation(model; Δt, stop_time=DateTime(2021, 2, 1))

    filepath = "test_datetime.nc"
    isfile(filepath) && rm(filepath)

    simulation.output_writers[:netcdf] =
        NetCDFOutputWriter(model, fields(model);
            filename = filepath,
            schedule = IterationInterval(1)
        )

    run!(simulation)

    ds = NCDataset(filepath)
    @test ds["time"].attrib["units"] == "seconds since 2000-01-01 00:00:00"

    Nt = length(ds["time"])
    @test Nt == 8 # There should be 8 outputs total.

    for n in 1:Nt-1
        @test ds["time"][n] == DateTime(2021, 1, 1) + (n-1) * Millisecond(1000Δt)
    end

    @test ds["time"][Nt] == DateTime(2021, 2, 1)

    close(ds)
    rm(filepath)

    return nothing
end

function test_timedate_netcdf_output(arch)
    grid = RectilinearGrid(arch, size=(1, 1, 1), extent=(1, 1, 1))

    clock = Clock(time=TimeDate(2021, 1, 1))

    model = NonhydrostaticModel(; grid, clock,
        timestepper = :QuasiAdamsBashforth2,
        buoyancy = SeawaterBuoyancy(),
        tracers = (:T, :S)
    )

    Δt = 5days + 3hours + 44.123seconds
    simulation = Simulation(model, Δt=Δt, stop_time=TimeDate(2021, 2, 1))

    filepath = "test_timedate.nc"
    isfile(filepath) && rm(filepath)

    simulation.output_writers[:netcdf] =
        NetCDFOutputWriter(model, fields(model);
            filename = filepath,
            schedule = IterationInterval(1)
        )

    run!(simulation)

    ds = NCDataset(filepath)
    @test ds["time"].attrib["units"] == "seconds since 2000-01-01 00:00:00"

    Nt = length(ds["time"])
    @test Nt == 8 # There should be 8 outputs total.

    for n in 1:Nt-1
        @test ds["time"][n] == DateTime(2021, 1, 1) + (n-1) * Millisecond(1000Δt)
    end

    @test ds["time"][Nt] == DateTime(2021, 2, 1)

    close(ds)
    rm(filepath)

    return nothing
end

function test_netcdf_size_file_splitting(arch)
    grid = RectilinearGrid(arch,
        size = (16, 16, 16),
        extent = (1, 1, 1),
        halo = (1, 1, 1)
    )

    model = NonhydrostaticModel(; grid,
        buoyancy = SeawaterBuoyancy(),
        tracers = (:T, :S)
    )

    simulation = Simulation(model, Δt=1, stop_iteration=10)

    fake_attributes = Dict("fake_attribute" => "fake_attribute")

    ow = NetCDFOutputWriter(model, (; u=model.velocities.u);
        dir = ".",
        filename = "test_size_file_splitting.nc",
        schedule = IterationInterval(1),
        array_type = Array{Float64},
        with_halos = true,
        global_attributes = fake_attributes,
        file_splitting = FileSizeLimit(200KiB),
        overwrite_existing = true
    )

    push!(simulation.output_writers, ow)

    # 531 KiB of output will be written which should get split into 3 files.
    run!(simulation)

    # Test that files has been split according to size as expected.
    @test filesize("test_size_file_splitting_part1.nc") > 200KiB
    @test filesize("test_size_file_splitting_part2.nc") > 200KiB
    @test filesize("test_size_file_splitting_part3.nc") < 200KiB
    @test !isfile("test_size_file_splitting_part4.nc")

    for n in string.(1:3)
        filename = "test_size_file_splitting_part$n.nc"
        ds = NCDataset(filename, "r")

        # Test that all files contain the same dimensions.
        num_dims = length(keys(ds.dim))
        @test num_dims == 7

        # Test that all files contain the user defined attributes.
        @test ds.attrib["fake_attribute"] == "fake_attribute"

        close(ds)

        # Leave test directory clean.
        rm(filename)
    end

    return nothing
end

function test_netcdf_time_file_splitting(arch)
    grid = RectilinearGrid(arch,
        size = (16, 16, 16),
        extent = (1, 1, 1),
        halo = (1, 1, 1)
    )

    model = NonhydrostaticModel(; grid,
        buoyancy = SeawaterBuoyancy(),
        tracers=(:T, :S)
    )

    simulation = Simulation(model, Δt=1, stop_time=12seconds)

    fake_attributes = Dict("fake_attribute" => "fake_attribute")

    ow = NetCDFOutputWriter(model, (; u=model.velocities.u);
        dir = ".",
        filename = "test_time_file_splitting.nc",
        schedule = IterationInterval(2),
        array_type = Array{Float64},
        with_halos = true,
        global_attributes = fake_attributes,
        file_splitting = TimeInterval(4seconds),
        overwrite_existing = true
    )

    push!(simulation.output_writers, ow)

    run!(simulation)

    for n in string.(1:3)
        filename = "test_time_file_splitting_part$n.nc"
        ds = NCDataset(filename, "r")

        # Test that all files contain the same dimensions.
        num_dims = length(ds["time"])
        @test num_dims == 2

        # Test that all files contain the user defined attributes.
        @test ds.attrib["fake_attribute"] == "fake_attribute"

        close(ds)

        # Leave test directory clean.
        rm(filename)
    end

    # TODO: Why is there a part 4? We should add more tests.
    rm("test_time_file_splitting_part4.nc")

    return nothing
end

function test_thermal_bubble_netcdf_output(arch)
    Nx, Ny, Nz = 16, 16, 16
    Lx, Ly, Lz = 100, 100, 100

    grid = RectilinearGrid(arch,
        topology = (Periodic, Periodic, Bounded),
        size = (Nx, Ny, Nz),
        extent = (Lx, Ly, Lz)
    )

    model = NonhydrostaticModel(; grid,
        closure = ScalarDiffusivity(ν=4e-2, κ=4e-2),
        buoyancy = SeawaterBuoyancy(),
        tracers = (:T, :S)
    )

    simulation = Simulation(model, Δt=6, stop_iteration=10)

    # Add a cube-shaped warm temperature anomaly that takes up the middle 50%
    # of the domain volume.
    i1, i2 = round(Int, Nx/4), round(Int, 3Nx/4)
    j1, j2 = round(Int, Ny/4), round(Int, 3Ny/4)
    k1, k2 = round(Int, Nz/4), round(Int, 3Nz/4)
    view(model.tracers.T, i1:i2, j1:j2, k1:k2) .+= 0.01

    outputs = Dict(
        "v" => model.velocities.v,
        "u" => model.velocities.u,
        "w" => model.velocities.w,
        "T" => model.tracers.T,
        "S" => model.tracers.S
    )

    nc_filepath = "test_thermal_bubble_$(typeof(arch)).nc"
    isfile(nc_filepath) && rm(nc_filepath)

    nc_writer = NetCDFOutputWriter(model, outputs,
        filename = nc_filepath,
        schedule = IterationInterval(10),
        verbose = true
    )

    push!(simulation.output_writers, nc_writer)

    i_slice = 1:10
    j_slice = 13
    k_slice = 9:11
    indices = (i_slice, j_slice, k_slice)
    j_slice = j_slice:j_slice  # So we can correctly index with it for later tests.

    nc_sliced_filepath = "test_thermal_bubble_sliced_$(typeof(arch)).nc"
    isfile(nc_sliced_filepath) && rm(nc_sliced_filepath)

    nc_sliced_writer = NetCDFOutputWriter(model, outputs,
        filename = nc_sliced_filepath,
        schedule = IterationInterval(10),
        array_type = Array{Float32},
        indices = indices,
        verbose = true
    )

    push!(simulation.output_writers, nc_sliced_writer)

    run!(simulation)

    ds3 = NCDataset(nc_filepath)

    @test haskey(ds3.attrib, "date")
    @test haskey(ds3.attrib, "Julia")
    @test haskey(ds3.attrib, "Oceananigans")
    @test haskey(ds3.attrib, "schedule")
    @test haskey(ds3.attrib, "interval")
    @test haskey(ds3.attrib, "output iteration interval")

    @test !isnothing(ds3.attrib["date"])
    @test !isnothing(ds3.attrib["Julia"])
    @test !isnothing(ds3.attrib["Oceananigans"])
    @test ds3.attrib["schedule"] == "IterationInterval"
    @test ds3.attrib["interval"] == 10
    @test !isnothing(ds3.attrib["output iteration interval"])

    @test eltype(ds3["time"]) == eltype(model.clock.time)

    @test eltype(ds3["x_c"]) == Float64
    @test eltype(ds3["x_f"]) == Float64
    @test eltype(ds3["y_c"]) == Float64
    @test eltype(ds3["y_f"]) == Float64
    @test eltype(ds3["z_c"]) == Float64
    @test eltype(ds3["z_f"]) == Float64

    @test length(ds3["x_c"]) == Nx
    @test length(ds3["y_c"]) == Ny
    @test length(ds3["z_c"]) == Nz
    @test length(ds3["x_f"]) == Nx
    @test length(ds3["y_f"]) == Ny
    @test length(ds3["z_f"]) == Nz+1  # z is Bounded

    @test ds3["x_c"][1] == grid.xᶜᵃᵃ[1]
    @test ds3["x_f"][1] == grid.xᶠᵃᵃ[1]
    @test ds3["y_c"][1] == grid.yᵃᶜᵃ[1]
    @test ds3["y_f"][1] == grid.yᵃᶠᵃ[1]
    @test ds3["z_c"][1] == grid.z.cᵃᵃᶜ[1]
    @test ds3["z_f"][1] == grid.z.cᵃᵃᶠ[1]

    @test ds3["x_c"][end] == grid.xᶜᵃᵃ[Nx]
    @test ds3["x_f"][end] == grid.xᶠᵃᵃ[Nx]
    @test ds3["y_c"][end] == grid.yᵃᶜᵃ[Ny]
    @test ds3["y_f"][end] == grid.yᵃᶠᵃ[Ny]
    @test ds3["z_c"][end] == grid.z.cᵃᵃᶜ[Nz]
    @test ds3["z_f"][end] == grid.z.cᵃᵃᶠ[Nz+1]  # z is Bounded

    @test eltype(ds3["u"]) == Float64
    @test eltype(ds3["v"]) == Float64
    @test eltype(ds3["w"]) == Float64
    @test eltype(ds3["T"]) == Float64
    @test eltype(ds3["S"]) == Float64

    u = ds3["u"][:, :, :, end]
    v = ds3["v"][:, :, :, end]
    w = ds3["w"][:, :, :, end]
    T = ds3["T"][:, :, :, end]
    S = ds3["S"][:, :, :, end]

    close(ds3)

    @test all(u .≈ Array(interior(model.velocities.u)))
    @test all(v .≈ Array(interior(model.velocities.v)))
    @test all(w .≈ Array(interior(model.velocities.w)))
    @test all(T .≈ Array(interior(model.tracers.T)))
    @test all(S .≈ Array(interior(model.tracers.S)))

    ds2 = NCDataset(nc_sliced_filepath)

    @test haskey(ds2.attrib, "date")
    @test haskey(ds2.attrib, "Julia")
    @test haskey(ds2.attrib, "Oceananigans")
    @test haskey(ds2.attrib, "schedule")
    @test haskey(ds2.attrib, "interval")
    @test haskey(ds2.attrib, "output iteration interval")

    @test !isnothing(ds2.attrib["date"])
    @test !isnothing(ds2.attrib["Julia"])
    @test !isnothing(ds2.attrib["Oceananigans"])
    @test ds2.attrib["schedule"] == "IterationInterval"
    @test ds2.attrib["interval"] == 10
    @test !isnothing(ds2.attrib["output iteration interval"])

    @test eltype(ds2["time"]) == eltype(model.clock.time)

    @test eltype(ds2["x_c"]) == Float32
    @test eltype(ds2["x_f"]) == Float32
    @test eltype(ds2["y_c"]) == Float32
    @test eltype(ds2["y_f"]) == Float32
    @test eltype(ds2["z_c"]) == Float32
    @test eltype(ds2["z_f"]) == Float32

    @test length(ds2["x_c"]) == length(i_slice)
    @test length(ds2["x_f"]) == length(i_slice)
    @test length(ds2["y_c"]) == length(j_slice)
    @test length(ds2["y_f"]) == length(j_slice)
    @test length(ds2["z_c"]) == length(k_slice)
    @test length(ds2["z_f"]) == length(k_slice)

    @test ds2["x_c"][1] == grid.xᶜᵃᵃ[i_slice[1]]
    @test ds2["x_f"][1] == grid.xᶠᵃᵃ[i_slice[1]]
    @test ds2["y_c"][1] == grid.yᵃᶜᵃ[j_slice[1]]
    @test ds2["y_f"][1] == grid.yᵃᶠᵃ[j_slice[1]]
    @test ds2["z_c"][1] == grid.z.cᵃᵃᶜ[k_slice[1]]
    @test ds2["z_f"][1] == grid.z.cᵃᵃᶠ[k_slice[1]]

    @test ds2["x_c"][end] == grid.xᶜᵃᵃ[i_slice[end]]
    @test ds2["x_f"][end] == grid.xᶠᵃᵃ[i_slice[end]]
    @test ds2["y_c"][end] == grid.yᵃᶜᵃ[j_slice[end]]
    @test ds2["y_f"][end] == grid.yᵃᶠᵃ[j_slice[end]]
    @test ds2["z_c"][end] == grid.z.cᵃᵃᶜ[k_slice[end]]
    @test ds2["z_f"][end] == grid.z.cᵃᵃᶠ[k_slice[end]]

    @test eltype(ds2["u"]) == Float32
    @test eltype(ds2["v"]) == Float32
    @test eltype(ds2["w"]) == Float32
    @test eltype(ds2["T"]) == Float32
    @test eltype(ds2["S"]) == Float32

    u_sliced = ds2["u"][:, :, :, end]
    v_sliced = ds2["v"][:, :, :, end]
    w_sliced = ds2["w"][:, :, :, end]
    T_sliced = ds2["T"][:, :, :, end]
    S_sliced = ds2["S"][:, :, :, end]

    close(ds2)

    @test all(u_sliced .≈ Array(interior(model.velocities.u))[i_slice, j_slice, k_slice])
    @test all(v_sliced .≈ Array(interior(model.velocities.v))[i_slice, j_slice, k_slice])
    @test all(w_sliced .≈ Array(interior(model.velocities.w))[i_slice, j_slice, k_slice])
    @test all(T_sliced .≈ Array(interior(model.tracers.T))[i_slice, j_slice, k_slice])
    @test all(S_sliced .≈ Array(interior(model.tracers.S))[i_slice, j_slice, k_slice])

    rm(nc_filepath)
    rm(nc_sliced_filepath)

    return nothing
end

function test_thermal_bubble_netcdf_output_with_halos(arch)
    Nx, Ny, Nz = 16, 16, 16
    Lx, Ly, Lz = 100, 100, 100
    Hx, Hy, Hz = 4, 3, 2

    grid = RectilinearGrid(arch,
        topology = (Periodic, Periodic, Bounded),
        size = (Nx, Ny, Nz),
        halo = (Hx, Hy, Hz),
        extent = (Lx, Ly, Lz),
    )

    model = NonhydrostaticModel(; grid,
        closure = ScalarDiffusivity(ν=4e-2, κ=4e-2),
        buoyancy = SeawaterBuoyancy(),
        tracers = (:T, :S)
    )

    simulation = Simulation(model, Δt=6, stop_iteration=10)

    # Add a cube-shaped warm temperature anomaly that takes up the middle 50%
    # of the domain volume.
    i1, i2 = round(Int, Nx/4), round(Int, 3Nx/4)
    j1, j2 = round(Int, Ny/4), round(Int, 3Ny/4)
    k1, k2 = round(Int, Nz/4), round(Int, 3Nz/4)
    view(model.tracers.T, i1:i2, j1:j2, k1:k2) .+= 0.01

    nc_filepath = "test_thermal_bubble_with_halos_$(typeof(arch)).nc"

    nc_writer = NetCDFOutputWriter(model,
        merge(model.velocities, model.tracers),
        filename = nc_filepath,
        schedule = IterationInterval(10),
        with_halos = true
    )

    push!(simulation.output_writers, nc_writer)

    run!(simulation)

    ds = NCDataset(nc_filepath)

    @test haskey(ds.attrib, "date")
    @test haskey(ds.attrib, "Julia")
    @test haskey(ds.attrib, "Oceananigans")
    @test haskey(ds.attrib, "schedule")
    @test haskey(ds.attrib, "interval")
    @test haskey(ds.attrib, "output iteration interval")

    @test !isnothing(ds.attrib["date"])
    @test !isnothing(ds.attrib["Julia"])
    @test !isnothing(ds.attrib["Oceananigans"])
    @test ds.attrib["schedule"] == "IterationInterval"
    @test ds.attrib["interval"] == 10
    @test !isnothing(ds.attrib["output iteration interval"])

    @test eltype(ds["time"]) == eltype(model.clock.time)

    # Using default array_type = Array{Float64}
    @test eltype(ds["x_c"]) == Float64
    @test eltype(ds["x_f"]) == Float64
    @test eltype(ds["y_c"]) == Float64
    @test eltype(ds["y_f"]) == Float64
    @test eltype(ds["z_c"]) == Float64
    @test eltype(ds["z_f"]) == Float64

    @test length(ds["x_c"]) == Nx+2Hx
    @test length(ds["y_c"]) == Ny+2Hy
    @test length(ds["z_c"]) == Nz+2Hz
    @test length(ds["x_f"]) == Nx+2Hx
    @test length(ds["y_f"]) == Ny+2Hy
    @test length(ds["z_f"]) == Nz+2Hz+1  # z is Bounded

    @test ds["x_c"][1] == grid.xᶜᵃᵃ[1-Hx]
    @test ds["x_f"][1] == grid.xᶠᵃᵃ[1-Hx]
    @test ds["y_c"][1] == grid.yᵃᶜᵃ[1-Hy]
    @test ds["y_f"][1] == grid.yᵃᶠᵃ[1-Hy]
    @test ds["z_c"][1] == grid.z.cᵃᵃᶜ[1-Hz]
    @test ds["z_f"][1] == grid.z.cᵃᵃᶠ[1-Hz]

    @test ds["x_c"][end] == grid.xᶜᵃᵃ[Nx+Hx]
    @test ds["x_f"][end] == grid.xᶠᵃᵃ[Nx+Hx]
    @test ds["y_c"][end] == grid.yᵃᶜᵃ[Ny+Hy]
    @test ds["y_f"][end] == grid.yᵃᶠᵃ[Ny+Hy]
    @test ds["z_c"][end] == grid.z.cᵃᵃᶜ[Nz+Hz]
    @test ds["z_f"][end] == grid.z.cᵃᵃᶠ[Nz+Hz+1]  # z is Bounded

    @test eltype(ds["u"]) == Float64
    @test eltype(ds["v"]) == Float64
    @test eltype(ds["w"]) == Float64
    @test eltype(ds["T"]) == Float64
    @test eltype(ds["S"]) == Float64

    u = ds["u"][:, :, :, end]
    v = ds["v"][:, :, :, end]
    w = ds["w"][:, :, :, end]
    T = ds["T"][:, :, :, end]
    S = ds["S"][:, :, :, end]

    close(ds)

    @test all(u .≈ Array(model.velocities.u.data.parent))
    @test all(v .≈ Array(model.velocities.v.data.parent))
    @test all(w .≈ Array(model.velocities.w.data.parent))
    @test all(T .≈ Array(model.tracers.T.data.parent))
    @test all(S .≈ Array(model.tracers.S.data.parent))

    rm(nc_filepath)

    return nothing
end

function test_netcdf_function_output(arch)
    Nx = Ny = Nz = N = 16
    L = 1
    Δt = 1.25
    iters = 3

    grid = RectilinearGrid(arch,
        size = (Nx, Ny, Nz),
        extent = (L, 2L, 3L)
    )

    model = NonhydrostaticModel(; grid,
        timestepper = :QuasiAdamsBashforth2,
        buoyancy = SeawaterBuoyancy(),
        tracers = (:T, :S)
    )

    simulation = Simulation(model, Δt=Δt, stop_iteration=iters)

    grid = model.grid # TODO: Figure out why?

    # Define scalar, vector, and 2D slice outputs
    f(model) = model.clock.time^2

    g(model) = model.clock.time .* exp.(znodes(grid, Center()))

    xC = xnodes(grid, Center())
    yF = ynodes(grid, Face())

    XC = [xC[i] for i in 1:Nx, j in 1:Ny]
    YF = [yF[j] for i in 1:Nx, j in 1:Ny]

    h(model) = @. model.clock.time * sin(XC) * cos(YF) # xy slice output

    outputs = (scalar=f, profile=g, slice=h)
    dims = (scalar=(), profile=("z_c",), slice=("x_c", "y_c"))

    output_attributes = (
        scalar = (long_name="Some scalar", units="bananas"),
        profile = (long_name="Some vertical profile", units="watermelons"),
        slice = (long_name="Some slice", units="mushrooms")
    )

    global_attributes = (
        location = "Bay of Fundy",
        onions = 7
    )

    nc_filepath = "test_function_outputs_$(typeof(arch)).nc"

    simulation.output_writers[:food] =
        NetCDFOutputWriter(model, outputs;
            global_attributes,
            output_attributes,
            filename = nc_filepath,
            schedule = TimeInterval(Δt),
            dimensions = dims,
            array_type = Array{Float64},
            verbose = true
        )

    run!(simulation)

    ds = NCDataset(nc_filepath, "r")

    @test haskey(ds.attrib, "date")
    @test haskey(ds.attrib, "Julia")
    @test haskey(ds.attrib, "Oceananigans")
    @test haskey(ds.attrib, "schedule")
    @test haskey(ds.attrib, "interval")
    @test haskey(ds.attrib, "output time interval")

    @test !isnothing(ds.attrib["date"])
    @test !isnothing(ds.attrib["Julia"])
    @test !isnothing(ds.attrib["Oceananigans"])
    @test !isnothing(ds.attrib["schedule"])
    @test !isnothing(ds.attrib["interval"])
    @test !isnothing(ds.attrib["output time interval"])

    @test eltype(ds["time"]) == eltype(model.clock.time)

    @test eltype(ds["x_c"]) == Float64
    @test eltype(ds["x_f"]) == Float64
    @test eltype(ds["y_c"]) == Float64
    @test eltype(ds["y_f"]) == Float64
    @test eltype(ds["z_c"]) == Float64
    @test eltype(ds["z_f"]) == Float64

    @test length(ds["x_c"]) == N
    @test length(ds["y_c"]) == N
    @test length(ds["z_c"]) == N
    @test length(ds["x_f"]) == N
    @test length(ds["y_f"]) == N
    @test length(ds["z_f"]) == N+1  # z is Bounded

    @test ds["x_c"][1] == grid.xᶜᵃᵃ[1]
    @test ds["x_f"][1] == grid.xᶠᵃᵃ[1]
    @test ds["y_c"][1] == grid.yᵃᶜᵃ[1]
    @test ds["y_f"][1] == grid.yᵃᶠᵃ[1]
    @test ds["z_c"][1] == grid.z.cᵃᵃᶜ[1]
    @test ds["z_f"][1] == grid.z.cᵃᵃᶠ[1]

    @test ds["x_c"][end] == grid.xᶜᵃᵃ[N]
    @test ds["y_c"][end] == grid.yᵃᶜᵃ[N]
    @test ds["x_f"][end] == grid.xᶠᵃᵃ[N]
    @test ds["y_f"][end] == grid.yᵃᶠᵃ[N]
    @test ds["z_c"][end] == grid.z.cᵃᵃᶜ[N]
    @test ds["z_f"][end] == grid.z.cᵃᵃᶠ[N+1]  # z is Bounded

    @test ds.attrib["location"] == "Bay of Fundy"
    @test ds.attrib["onions"] == 7

    @test eltype(ds["scalar"]) == Float64
    @test eltype(ds["profile"]) == Float64
    @test eltype(ds["slice"]) == Float64

    @test length(ds["time"]) == iters+1
    @test ds["time"][:] == [n*Δt for n in 0:iters]

    @test length(ds["scalar"]) == iters+1
    @test ds["scalar"].attrib["long_name"] == "Some scalar"
    @test ds["scalar"].attrib["units"] == "bananas"
    @test ds["scalar"][:] == [(n*Δt)^2 for n in 0:iters]
    @test dimnames(ds["scalar"]) == ("time",)

    @test ds["profile"].attrib["long_name"] == "Some vertical profile"
    @test ds["profile"].attrib["units"] == "watermelons"
    @test size(ds["profile"]) == (N, iters+1)
    @test dimnames(ds["profile"]) == ("z_c", "time")

    for n in 0:iters
        @test ds["profile"][:, n+1] == n*Δt .* exp.(znodes(grid, Center()))
    end

    @test ds["slice"].attrib["long_name"] == "Some slice"
    @test ds["slice"].attrib["units"] == "mushrooms"
    @test size(ds["slice"]) == (N, N, iters+1)
    @test dimnames(ds["slice"]) == ("x_c", "y_c", "time")

    for n in 0:iters
        @test ds["slice"][:, :, n+1] == n*Δt .* sin.(XC) .* cos.(YF)
    end

    close(ds)

    #####
    ##### Take 1 more time step and test that appending to a NetCDF file works
    #####

    iters += 1
    simulation = Simulation(model, Δt=Δt, stop_iteration=iters)

    simulation.output_writers[:food] =
        NetCDFOutputWriter(model, outputs;
            global_attributes,
            output_attributes,
            filename = nc_filepath,
            overwrite_existing = false,
            schedule = IterationInterval(1),
            array_type = Array{Float64},
            dimensions = dims,
            verbose = true
        )

    run!(simulation)

    ds = NCDataset(nc_filepath, "r")

    @test length(ds["time"]) == iters+1
    @test length(ds["scalar"]) == iters+1
    @test size(ds["profile"]) == (N, iters+1)
    @test size(ds["slice"]) == (N, N, iters+1)

    @test ds["time"][:] == [n*Δt for n in 0:iters]
    @test ds["scalar"][:] == [(n*Δt)^2 for n in 0:iters]

    for n in 0:iters
        @test ds["profile"][:, n+1] ≈ n*Δt .* exp.(znodes(grid, Center()))
        @test ds["slice"][:, :, n+1] ≈ n*Δt .* (sin.(XC) .* cos.(YF))
    end

    close(ds)

    rm(nc_filepath)

    return nothing
end

function test_netcdf_spatial_average(arch)
    topo = (Periodic, Periodic, Periodic)
    domain = (x=(0, 1), y=(0, 1), z=(0, 1))
    grid = RectilinearGrid(arch, topology=topo, size=(4, 4, 4); domain...)

    model = NonhydrostaticModel(; grid,
        timestepper = :RungeKutta3,
        tracers = (:c,),
        coriolis = nothing,
        buoyancy = nothing,
        closure = nothing
    )

    set!(model, c=1)

    Δt = 0.01 # Floating point number chosen conservatively to flag rounding errors

    simulation = Simulation(model, Δt=Δt, stop_iteration=10)

    ∫c_dx = Field(Average(model.tracers.c, dims=(1)))
    ∫∫c_dxdy = Field(Average(model.tracers.c, dims=(1, 2)))
    ∫∫∫c_dxdydz = Field(Average(model.tracers.c, dims=(1, 2, 3)))

    nc_filepath = "volume_averaged_field_test.nc"

    simulation.output_writers[:averages] =
        NetCDFOutputWriter(model,
            (; ∫c_dx, ∫∫c_dxdy, ∫∫∫c_dxdydz),
            array_type = Array{Float64},
            verbose = true,
            filename = nc_filepath,
            schedule = IterationInterval(2)
        )

    run!(simulation)

    ds = NCDataset(nc_filepath)

    for (n, t) in enumerate(ds["time"])
        @test all(ds["∫c_dx"][:,:, n] .≈ 1)
        @test all(ds["∫∫c_dxdy"][:, n] .≈ 1)
        @test all(ds["∫∫∫c_dxdydz"][n] .≈ 1)
    end

    close(ds)

    return nothing
end


function test_netcdf_time_averaging(arch)
    # Test for both "nice" floating point number and one that is more susceptible
    # to rounding errors
    for Δt in (1/64, 0.01)
        # Results should be very close (rtol < 1e-5) for stride = 1.
        # stride > 2 is currently not robust and can give inconsistent
        # results due to floating number errors that can result in vanishingly
        # small timesteps, which essentially decouples the clock time from
        # the iteration number.
        # Can add stride > 1 cases to the following line to test them.
        for (stride, rtol) in zip((1), (1e-5))
            @info "  Testing time-averaging of NetCDF outputs [$(typeof(arch))] with " *
                  "timestep of $(Δt), stride of $(stride), and relative tolerance of $(rtol)."

            topo = (Periodic, Periodic, Periodic)
            domain = (x=(0, 1), y=(0, 1), z=(0, 1))
            grid = RectilinearGrid(arch, topology=topo, size=(4, 4, 4); domain...)

            λ1(x, y, z) = x + (1 - y)^2 + tanh(z)
            λ2(x, y, z) = x + (1 - y)^2 + tanh(4z)

            Fc1(x, y, z, t, c1) = - λ1(x, y, z) * c1
            Fc2(x, y, z, t, c2) = - λ2(x, y, z) * c2

            c1_forcing = Forcing(Fc1, field_dependencies=:c1)
            c2_forcing = Forcing(Fc2, field_dependencies=:c2)

            model = NonhydrostaticModel(; grid,
                timestepper = :RungeKutta3,
                tracers = (:c1, :c2),
                forcing = (c1=c1_forcing, c2=c2_forcing)
            )

            set!(model, c1=1, c2=1)

            # Floating point number chosen conservatively to flag rounding errors
            simulation = Simulation(model, Δt=Δt, stop_time=50Δt)

            ∫c1_dxdy = Field(Average(model.tracers.c1, dims=(1, 2)))
            ∫c2_dxdy = Field(Average(model.tracers.c2, dims=(1, 2)))

            nc_outputs = Dict(
                "c1" => ∫c1_dxdy,
                "c2" => ∫c2_dxdy
            )

            nc_dimensions = Dict(
                "c1" => ("z_c",),
                "c2" => ("z_c",)
            )

            horizontal_average_nc_filepath = "decay_averaged_field_test.nc"

            simulation.output_writers[:horizontal_average] =
                NetCDFOutputWriter(
                    model,
                    nc_outputs,
                    array_type = Array{Float64},
                    verbose = true,
                    filename = horizontal_average_nc_filepath,
                    schedule = TimeInterval(10Δt),
                    dimensions = nc_dimensions
                )

            multiple_time_average_nc_filepath = "decay_windowed_time_average_test.nc"
            single_time_average_nc_filepath = "single_decay_windowed_time_average_test.nc"
            window = 6Δt

            single_nc_output = Dict("c1" => ∫c1_dxdy)
            single_nc_dimension = Dict("c1" => ("z_c",))

            simulation.output_writers[:single_output_time_average] =
                NetCDFOutputWriter(
                    model,
                    single_nc_output,
                    array_type = Array{Float64},
                    verbose = true,
                    filename = single_time_average_nc_filepath,
                    schedule = AveragedTimeInterval(10Δt, window = window, stride = stride),
                    dimensions = single_nc_dimension
                )

            simulation.output_writers[:multiple_output_time_average] =
                NetCDFOutputWriter(
                    model,
                    nc_outputs,
                    array_type = Array{Float64},
                    verbose = true,
                    filename = multiple_time_average_nc_filepath,
                    schedule = AveragedTimeInterval(10Δt, window = window, stride = stride),
                    dimensions = nc_dimensions
                )

            run!(simulation)

            ##### For each λ, the horizontal average should evaluate to
            #####
            #####     c̄(z, t) = ∫₀¹ ∫₀¹ exp{- λ(x, y, z) * t} dx dy
            #####             = 1 / (Nx*Ny) * Σᵢ₌₁ᴺˣ Σⱼ₌₁ᴺʸ exp{- λ(i, j, k) * t}
            #####
            ##### which we can compute analytically.

            ds = NCDataset(horizontal_average_nc_filepath)

            Nx, Ny, Nz = size(grid)
            xs, ys, zs = nodes(model.tracers.c1)

            c̄1(z, t) = 1 / (Nx * Ny) * sum(exp(-λ1(x, y, z) * t) for x in xs for y in ys)
            c̄2(z, t) = 1 / (Nx * Ny) * sum(exp(-λ2(x, y, z) * t) for x in xs for y in ys)

            for (n, t) in enumerate(ds["time"])
                @test all(isapprox.(ds["c1"][:, n], c̄1.(zs, t), rtol=rtol))
                @test all(isapprox.(ds["c2"][:, n], c̄2.(zs, t), rtol=rtol))
            end

            close(ds)

            # Compute time averages...
            c̄1(ts) = 1/length(ts) * sum(c̄1.(zs, t) for t in ts)
            c̄2(ts) = 1/length(ts) * sum(c̄2.(zs, t) for t in ts)

            #####
            ##### Test strided windowed time average against analytic solution
            ##### for *single* NetCDF output
            #####

            single_ds = NCDataset(single_time_average_nc_filepath)

            attribute_names = (
                "schedule", "interval", "output time interval",
                "time_averaging_window", "time averaging window",
                "time_averaging_stride", "time averaging stride"
            )

            for name in attribute_names
                @test haskey(single_ds.attrib, name)
                @test !isnothing(single_ds.attrib[name])
            end

            window_size = Int(window/Δt)

            @info "    Testing time-averaging of a single NetCDF output [$(typeof(arch))]..."

            for (n, t) in enumerate(single_ds["time"][2:end])
                averaging_times = [t - n*Δt for n in 0:stride:window_size-1 if t - n*Δt >= 0]
                @test all(isapprox.(single_ds["c1"][:, n+1], c̄1(averaging_times), rtol=rtol, atol=rtol))
            end

            close(single_ds)

            #####
            ##### Test strided windowed time average against analytic solution
            ##### for *multiple* NetCDF outputs
            #####

            ds = NCDataset(multiple_time_average_nc_filepath)

            @info "    Testing time-averaging of multiple NetCDF outputs [$(typeof(arch))]..."

            for (n, t) in enumerate(ds["time"][2:end])
                averaging_times = [t - n*Δt for n in 0:stride:window_size-1 if t - n*Δt >= 0]
                @test all(isapprox.(ds["c2"][:, n+1], c̄2(averaging_times), rtol=rtol))
            end

            close(ds)

            rm(horizontal_average_nc_filepath)
            rm(single_time_average_nc_filepath)
            rm(multiple_time_average_nc_filepath)
        end
    end

    return nothing
end

function test_netcdf_output_alignment(arch)
    grid = RectilinearGrid(size=(1, 1, 1), extent=(1, 1, 1))

    model = NonhydrostaticModel(; grid,
        timestepper = :QuasiAdamsBashforth2,
        buoyancy = SeawaterBuoyancy(),
        tracers = (:T, :S)
    )

    simulation = Simulation(model, Δt=0.2, stop_time=40)

    test_filename1 = "test_output_alignment1.nc"
    simulation.output_writers[:stuff] =
        NetCDFOutputWriter(model,
            model.velocities,
            filename = test_filename1,
            schedule = TimeInterval(7.3)
        )

    test_filename2 = "test_output_alignment2.nc"
    simulation.output_writers[:something] =
        NetCDFOutputWriter(model,
            model.tracers,
            filename = test_filename2,
            schedule = TimeInterval(3.0)
        )

    run!(simulation)

    NCDataset(test_filename1, "r") do ds
        @test all(ds["time"] .== 0:7.3:40)
    end

    NCDataset(test_filename2, "r") do ds
        @test all(ds["time"] .== 0:3.0:40)
    end

    rm(test_filename1)
    rm(test_filename2)

    return nothing
end

function test_netcdf_vertically_stretched_grid_output(arch)
    Nx = Ny = 8
    Nz = 16

    z_faces = [k^2 for k in 0:Nz]

    grid = RectilinearGrid(arch;
        size = (Nx, Ny, Nz),
        x = (0, 1),
        y = (-π, π),
        z = z_faces
    )

    model = NonhydrostaticModel(; grid,
        buoyancy = SeawaterBuoyancy(),
        tracers = (:T, :S)
    )

    simulation = Simulation(model, Δt=1.25, stop_iteration=3)

    nc_filepath = "test_netcdf_vertically_stretched_grid_output_$(typeof(arch)).nc"

    simulation.output_writers[:fields] =
        NetCDFOutputWriter(model,
            merge(model.velocities, model.tracers),
            filename = nc_filepath,
            schedule = IterationInterval(1),
            array_type = Array{Float64},
            verbose = true
        )

    run!(simulation)

    grid = model.grid # TODO: Again, figure out why?

    ds = NCDataset(nc_filepath)

    @test length(ds["x_c"]) == Nx
    @test length(ds["y_c"]) == Ny
    @test length(ds["z_c"]) == Nz
    @test length(ds["x_f"]) == Nx
    @test length(ds["y_f"]) == Ny
    @test length(ds["z_f"]) == Nz+1  # z is Bounded

    @test ds["x_c"][1] == grid.xᶜᵃᵃ[1]
    @test ds["x_f"][1] == grid.xᶠᵃᵃ[1]
    @test ds["y_c"][1] == grid.yᵃᶜᵃ[1]
    @test ds["y_f"][1] == grid.yᵃᶠᵃ[1]

    @test CUDA.@allowscalar ds["z_c"][1] == grid.z.cᵃᵃᶜ[1]
    @test CUDA.@allowscalar ds["z_f"][1] == grid.z.cᵃᵃᶠ[1]

    @test ds["x_c"][end] == grid.xᶜᵃᵃ[Nx]
    @test ds["x_f"][end] == grid.xᶠᵃᵃ[Nx]
    @test ds["y_c"][end] == grid.yᵃᶜᵃ[Ny]
    @test ds["y_f"][end] == grid.yᵃᶠᵃ[Ny]

    @test CUDA.@allowscalar ds["z_c"][end] == grid.z.cᵃᵃᶜ[Nz]
    @test CUDA.@allowscalar ds["z_f"][end] == grid.z.cᵃᵃᶠ[Nz+1]  # z is Bounded

    close(ds)
    rm(nc_filepath)

    return nothing
end

function test_netcdf_regular_lat_lon_grid_output(arch; immersed = false)
    Nλ = Nφ = Nz = 16

    grid = LatitudeLongitudeGrid(arch;
        size = (Nλ, Nφ, Nz),
        longitude = (-180, 180),
        latitude = (-80, 80),
        z = (-100, 0)
    )

    if immersed
        grid = ImmersedBoundaryGrid(grid, GridFittedBottom((x, y) -> -50))
    end

    model = HydrostaticFreeSurfaceModel(; grid, momentum_advection = VectorInvariant())

    simulation = Simulation(model, Δt=1.25, stop_iteration=3)

    nc_filepath = "test_netcdf_regular_lat_lon_grid_output_$(typeof(arch)).nc"

    simulation.output_writers[:fields] =
        NetCDFOutputWriter(model,
            merge(model.velocities, model.tracers),
            filename = nc_filepath,
            schedule = IterationInterval(1),
            array_type = Array{Float64},
            verbose = true
        )

    run!(simulation)

    grid = model.grid

    ds = NCDataset(nc_filepath)

    @test length(ds["x_c"]) == Nλ
    @test length(ds["y_c"]) == Nφ
    @test length(ds["z_c"]) == Nz
    @test length(ds["x_f"]) == Nλ
    @test length(ds["y_f"]) == Nφ+1  # y is Bounded
    @test length(ds["z_f"]) == Nz+1  # z is Bounded

    @test ds["x_c"][1] == grid.λᶜᵃᵃ[1]
    @test ds["x_f"][1] == grid.λᶠᵃᵃ[1]
    @test ds["y_c"][1] == grid.φᵃᶜᵃ[1]
    @test ds["y_f"][1] == grid.φᵃᶠᵃ[1]
    @test ds["z_c"][1] == grid.z.cᵃᵃᶜ[1]
    @test ds["z_f"][1] == grid.z.cᵃᵃᶠ[1]

    @test ds["x_c"][end] == grid.λᶜᵃᵃ[Nλ]
    @test ds["x_f"][end] == grid.λᶠᵃᵃ[Nλ]
    @test ds["y_c"][end] == grid.φᵃᶜᵃ[Nφ]
    @test ds["y_f"][end] == grid.φᵃᶠᵃ[Nφ+1]  # y is Bounded
    @test ds["z_c"][end] == grid.z.cᵃᵃᶜ[Nz]
    @test ds["z_f"][end] == grid.z.cᵃᵃᶠ[Nz+1]  # z is Bounded

    close(ds)
    rm(nc_filepath)

    return nothing
end

for arch in archs
    @testset "NetCDF output writer [$(typeof(arch))]" begin
        @info "  Testing NetCDF output writer [$(typeof(arch))]..."

        test_datetime_netcdf_output(arch)
        test_timedate_netcdf_output(arch)

        test_netcdf_size_file_splitting(arch)
        test_netcdf_time_file_splitting(arch)

        test_thermal_bubble_netcdf_output(arch)
        test_thermal_bubble_netcdf_output_with_halos(arch)

        test_netcdf_function_output(arch)
        test_netcdf_output_alignment(arch)

        test_netcdf_spatial_average(arch)
        # test_netcdf_time_averaging(arch)

        test_netcdf_vertically_stretched_grid_output(arch)

        # test_netcdf_regular_lat_lon_grid_output(arch; immersed = false)
        # test_netcdf_regular_lat_lon_grid_output(arch; immersed = true)
    end
end
