using Statistics
using NCDatasets

using Oceananigans.Diagnostics
using Oceananigans.Fields
using Oceananigans.OutputWriters

using Dates: Millisecond
using Oceananigans.BoundaryConditions: PBC, FBC, ZFBC, ContinuousBoundaryFunction
using Oceananigans.TimeSteppers: update_state!

function instantiate_windowed_time_average(model)

    set!(model, u = (x, y, z) -> rand())

    u, v, w = model.velocities

    u₀ = similar(interior(u))
    u₀ .= interior(u)

    wta = WindowedTimeAverage(model.velocities.u, schedule=AveragedTimeInterval(10, window=1))

    return all(wta(model) .== u₀)
end

function time_step_with_windowed_time_average(model)

    model.clock.iteration = 0
    model.clock.time = 0.0

    set!(model, u=0, v=0, w=0, T=0, S=0)

    wta = WindowedTimeAverage(model.velocities.u, schedule=AveragedTimeInterval(4, window=2))

    simulation = Simulation(model, Δt=1.0, stop_time=4.0)
    simulation.diagnostics[:u_avg] = wta
    run!(simulation)

    return all(wta(model) .== interior(model.velocities.u))
end

#####
##### NetCDFOutputWriter tests
#####

function run_DateTime_netcdf_tests(arch)
    grid = RegularRectilinearGrid(size=(1, 1, 1), extent=(1, 1, 1))
    clock = Clock(time=DateTime(2021, 1, 1))
    model = IncompressibleModel(architecture=arch, grid=grid, clock=clock)

    Δt = 5days + 3hours + 44.123seconds
    simulation = Simulation(model, Δt=Δt, stop_time=DateTime(2021, 2, 1))

    filepath = "test_DateTime.nc"
    simulation.output_writers[:cal] = NetCDFOutputWriter(model, fields(model), filepath=filepath, schedule=IterationInterval(1))

    run!(simulation)

    ds = NCDataset(filepath)
    @test ds["time"].attrib["units"] == "seconds since 2000-01-01 00:00:00"

    Nt = length(ds["time"])
    @test Nt == 8

    for n in 1:Nt-1
        @test ds["time"][n] == DateTime(2021, 1, 1) + (n-1) * Millisecond(1000Δt)
    end

    @test ds["time"][Nt] == DateTime(2021, 2, 1)

    close(ds)
    rm(filepath)

    return nothing
end

function run_TimeDate_netcdf_tests(arch)
    grid = RegularRectilinearGrid(size=(1, 1, 1), extent=(1, 1, 1))
    clock = Clock(time=TimeDate(2021, 1, 1))
    model = IncompressibleModel(architecture=arch, grid=grid, clock=clock)

    Δt = 5days + 3hours + 44.123seconds
    simulation = Simulation(model, Δt=Δt, stop_time=TimeDate(2021, 2, 1))

    filepath = "test_TimeDate.nc"
    simulation.output_writers[:cal] = NetCDFOutputWriter(model, fields(model), filepath=filepath, schedule=IterationInterval(1))

    run!(simulation)

    ds = NCDataset(filepath)
    @test ds["time"].attrib["units"] == "seconds since 2000-01-01 00:00:00"

    Nt = length(ds["time"])
    @test Nt == 8

    for n in 1:Nt-1
        @test ds["time"][n] == DateTime(2021, 1, 1) + (n-1) * Millisecond(1000Δt)
    end

    @test ds["time"][Nt] == DateTime(2021, 2, 1)

    close(ds)
    rm(filepath)

    return nothing
end

function run_thermal_bubble_netcdf_tests(arch)
    Nx, Ny, Nz = 16, 16, 16
    Lx, Ly, Lz = 100, 100, 100

    topo = (Periodic, Periodic, Bounded)
    grid = RegularRectilinearGrid(topology=topo, size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz))
    closure = IsotropicDiffusivity(ν=4e-2, κ=4e-2)
    model = IncompressibleModel(architecture=arch, grid=grid, closure=closure)
    simulation = Simulation(model, Δt=6, stop_iteration=10)

    # Add a cube-shaped warm temperature anomaly that takes up the middle 50%
    # of the domain volume.
    i1, i2 = round(Int, Nx/4), round(Int, 3Nx/4)
    j1, j2 = round(Int, Ny/4), round(Int, 3Ny/4)
    k1, k2 = round(Int, Nz/4), round(Int, 3Nz/4)
    CUDA.@allowscalar model.tracers.T.data[i1:i2, j1:j2, k1:k2] .+= 0.01

    outputs = Dict("v" => model.velocities.v,
                   "u" => model.velocities.u,
                   "w" => model.velocities.w,
                   "T" => model.tracers.T,
                   "S" => model.tracers.S)

    nc_filepath = "test_dump_$(typeof(arch)).nc"
    nc_writer = NetCDFOutputWriter(model, outputs, filepath=nc_filepath, schedule=IterationInterval(10), verbose=true)
    push!(simulation.output_writers, nc_writer)

    i_slice = 1:10
    j_slice = 13
    k_slice = 9:11
    field_slicer = FieldSlicer(i=i_slice, j=j_slice, k=k_slice)
    j_slice = j_slice:j_slice  # So we can correctly index with it for later tests.

    nc_sliced_filepath = "test_dump_sliced_$(typeof(arch)).nc"
    nc_sliced_writer = NetCDFOutputWriter(model, outputs, filepath=nc_sliced_filepath, schedule=IterationInterval(10),
                                          field_slicer=field_slicer, verbose=true)

    push!(simulation.output_writers, nc_sliced_writer)

    run!(simulation)

    ds3 = Dataset(nc_filepath)

    @test haskey(ds3.attrib, "date") && !isnothing(ds3.attrib["date"])
    @test haskey(ds3.attrib, "Julia") && !isnothing(ds3.attrib["Julia"])
    @test haskey(ds3.attrib, "Oceananigans") && !isnothing(ds3.attrib["Oceananigans"])
    @test haskey(ds3.attrib, "schedule") && ds3.attrib["schedule"] == "IterationInterval"
    @test haskey(ds3.attrib, "interval") && ds3.attrib["interval"] == 10
    @test haskey(ds3.attrib, "output iteration interval") && !isnothing(ds3.attrib["output iteration interval"])

    @test eltype(ds3["time"]) == eltype(model.clock.time)

    @test eltype(ds3["xC"]) == Float64
    @test eltype(ds3["xF"]) == Float64
    @test eltype(ds3["yC"]) == Float64
    @test eltype(ds3["yF"]) == Float64
    @test eltype(ds3["zC"]) == Float64
    @test eltype(ds3["zF"]) == Float64

    @test length(ds3["xC"]) == Nx
    @test length(ds3["yC"]) == Ny
    @test length(ds3["zC"]) == Nz
    @test length(ds3["xF"]) == Nx
    @test length(ds3["yF"]) == Ny
    @test length(ds3["zF"]) == Nz+1  # z is Bounded

    @test ds3["xC"][1] == grid.xC[1]
    @test ds3["xF"][1] == grid.xF[1]
    @test ds3["yC"][1] == grid.yC[1]
    @test ds3["yF"][1] == grid.yF[1]
    @test ds3["zC"][1] == grid.zC[1]
    @test ds3["zF"][1] == grid.zF[1]

    @test ds3["xC"][end] == grid.xC[Nx]
    @test ds3["xF"][end] == grid.xF[Nx]
    @test ds3["yC"][end] == grid.yC[Ny]
    @test ds3["yF"][end] == grid.yF[Ny]
    @test ds3["zC"][end] == grid.zC[Nz]
    @test ds3["zF"][end] == grid.zF[Nz+1]  # z is Bounded

    @test eltype(ds3["u"]) == Float32
    @test eltype(ds3["v"]) == Float32
    @test eltype(ds3["w"]) == Float32
    @test eltype(ds3["T"]) == Float32
    @test eltype(ds3["S"]) == Float32

    u = ds3["u"][:, :, :, end]
    v = ds3["v"][:, :, :, end]
    w = ds3["w"][:, :, :, end]
    T = ds3["T"][:, :, :, end]
    S = ds3["S"][:, :, :, end]

    close(ds3)

    @test all(u .≈ Array(interiorparent(model.velocities.u)))
    @test all(v .≈ Array(interiorparent(model.velocities.v)))
    @test all(w .≈ Array(interiorparent(model.velocities.w)))
    @test all(T .≈ Array(interiorparent(model.tracers.T)))
    @test all(S .≈ Array(interiorparent(model.tracers.S)))

    ds2 = Dataset(nc_sliced_filepath)

    @test haskey(ds2.attrib, "date") && !isnothing(ds2.attrib["date"])
    @test haskey(ds2.attrib, "Julia") && !isnothing(ds2.attrib["Julia"])
    @test haskey(ds2.attrib, "Oceananigans") && !isnothing(ds2.attrib["Oceananigans"])
    @test haskey(ds2.attrib, "schedule") && ds2.attrib["schedule"] == "IterationInterval"
    @test haskey(ds2.attrib, "interval") && ds2.attrib["interval"] == 10
    @test haskey(ds2.attrib, "output iteration interval") && !isnothing(ds2.attrib["output iteration interval"])

    @test eltype(ds2["time"]) == eltype(model.clock.time)

    @test eltype(ds2["xC"]) == Float64
    @test eltype(ds2["xF"]) == Float64
    @test eltype(ds2["yC"]) == Float64
    @test eltype(ds2["yF"]) == Float64
    @test eltype(ds2["zC"]) == Float64
    @test eltype(ds2["zF"]) == Float64

    @test length(ds2["xC"]) == length(i_slice)
    @test length(ds2["xF"]) == length(i_slice)
    @test length(ds2["yC"]) == length(j_slice)
    @test length(ds2["yF"]) == length(j_slice)
    @test length(ds2["zC"]) == length(k_slice)
    @test length(ds2["zF"]) == length(k_slice)

    @test ds2["xC"][1] == grid.xC[i_slice[1]]
    @test ds2["xF"][1] == grid.xF[i_slice[1]]
    @test ds2["yC"][1] == grid.yC[j_slice[1]]
    @test ds2["yF"][1] == grid.yF[j_slice[1]]
    @test ds2["zC"][1] == grid.zC[k_slice[1]]
    @test ds2["zF"][1] == grid.zF[k_slice[1]]

    @test ds2["xC"][end] == grid.xC[i_slice[end]]
    @test ds2["xF"][end] == grid.xF[i_slice[end]]
    @test ds2["yC"][end] == grid.yC[j_slice[end]]
    @test ds2["yF"][end] == grid.yF[j_slice[end]]
    @test ds2["zC"][end] == grid.zC[k_slice[end]]
    @test ds2["zF"][end] == grid.zF[k_slice[end]]

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

    @test all(u_sliced .≈ Array(interiorparent(model.velocities.u))[i_slice, j_slice, k_slice])
    @test all(v_sliced .≈ Array(interiorparent(model.velocities.v))[i_slice, j_slice, k_slice])
    @test all(w_sliced .≈ Array(interiorparent(model.velocities.w))[i_slice, j_slice, k_slice])
    @test all(T_sliced .≈ Array(interiorparent(model.tracers.T))[i_slice, j_slice, k_slice])
    @test all(S_sliced .≈ Array(interiorparent(model.tracers.S))[i_slice, j_slice, k_slice])

    rm(nc_filepath)
    rm(nc_sliced_filepath)

    return nothing
end

function run_thermal_bubble_netcdf_tests_with_halos(arch)
    Nx, Ny, Nz = 16, 16, 16
    Lx, Ly, Lz = 100, 100, 100

    topo = (Periodic, Periodic, Bounded)
    grid = RegularRectilinearGrid(topology=topo, size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz))
    closure = IsotropicDiffusivity(ν=4e-2, κ=4e-2)
    model = IncompressibleModel(architecture=arch, grid=grid, closure=closure)
    simulation = Simulation(model, Δt=6, stop_iteration=10)

    # Add a cube-shaped warm temperature anomaly that takes up the middle 50%
    # of the domain volume.
    i1, i2 = round(Int, Nx/4), round(Int, 3Nx/4)
    j1, j2 = round(Int, Ny/4), round(Int, 3Ny/4)
    k1, k2 = round(Int, Nz/4), round(Int, 3Nz/4)
    CUDA.@allowscalar model.tracers.T.data[i1:i2, j1:j2, k1:k2] .+= 0.01

    nc_filepath = "test_dump_with_halos_$(typeof(arch)).nc"
    nc_writer = NetCDFOutputWriter(model, merge(model.velocities, model.tracers),
                                   filepath=nc_filepath,
                                   schedule=IterationInterval(10),
                                   field_slicer=FieldSlicer(with_halos=true))

    push!(simulation.output_writers, nc_writer)

    run!(simulation)

    ds = Dataset(nc_filepath)

    @test haskey(ds.attrib, "date") && !isnothing(ds.attrib["date"])
    @test haskey(ds.attrib, "Julia") && !isnothing(ds.attrib["Julia"])
    @test haskey(ds.attrib, "Oceananigans") && !isnothing(ds.attrib["Oceananigans"])
    @test haskey(ds.attrib, "schedule") && ds.attrib["schedule"] == "IterationInterval"
    @test haskey(ds.attrib, "interval") && ds.attrib["interval"] == 10
    @test haskey(ds.attrib, "output iteration interval") && !isnothing(ds.attrib["output iteration interval"])

    @test eltype(ds["time"]) == eltype(model.clock.time)

    @test eltype(ds["xC"]) == Float64
    @test eltype(ds["xF"]) == Float64
    @test eltype(ds["yC"]) == Float64
    @test eltype(ds["yF"]) == Float64
    @test eltype(ds["zC"]) == Float64
    @test eltype(ds["zF"]) == Float64

    Hx, Hy, Hz = grid.Hx, grid.Hy, grid.Hz
    @test length(ds["xC"]) == Nx+2Hx
    @test length(ds["yC"]) == Ny+2Hy
    @test length(ds["zC"]) == Nz+2Hz
    @test length(ds["xF"]) == Nx+2Hx
    @test length(ds["yF"]) == Ny+2Hy
    @test length(ds["zF"]) == Nz+2Hz+1  # z is Bounded

    @test ds["xC"][1] == grid.xC[1-Hx]
    @test ds["xF"][1] == grid.xF[1-Hx]
    @test ds["yC"][1] == grid.yC[1-Hy]
    @test ds["yF"][1] == grid.yF[1-Hy]
    @test ds["zC"][1] == grid.zC[1-Hz]
    @test ds["zF"][1] == grid.zF[1-Hz]

    @test ds["xC"][end] == grid.xC[Nx+Hx]
    @test ds["xF"][end] == grid.xF[Nx+Hx]
    @test ds["yC"][end] == grid.yC[Ny+Hy]
    @test ds["yF"][end] == grid.yF[Ny+Hy]
    @test ds["zC"][end] == grid.zC[Nz+Hz]
    @test ds["zF"][end] == grid.zF[Nz+Hz+1]  # z is Bounded

    @test eltype(ds["u"]) == Float32
    @test eltype(ds["v"]) == Float32
    @test eltype(ds["w"]) == Float32
    @test eltype(ds["T"]) == Float32
    @test eltype(ds["S"]) == Float32

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

function run_netcdf_function_output_tests(arch)
    N = 16
    L = 1
    Δt = 1.25
    iters = 3

    grid = RegularRectilinearGrid(size=(N, N, N), extent=(L, 2L, 3L))
    model = IncompressibleModel(architecture=arch, grid=grid)
    simulation = Simulation(model, Δt=Δt, stop_iteration=iters)
    grid = model.grid

    # Define scalar, vector, and 2D slice outputs
    f(model) = model.clock.time^2

    g(model) = model.clock.time .* exp.(znodes(Center, grid))

    h(model) = model.clock.time .* (   sin.(xnodes(Center, grid, reshape=true)[:, :, 1])
                                    .* cos.(ynodes(Face, grid, reshape=true)[:, :, 1]))

    outputs = (scalar=f, profile=g, slice=h)
    dims = (scalar=(), profile=("zC",), slice=("xC", "yC"))

    output_attributes = (
        scalar = (longname="Some scalar", units="bananas"),
        profile = (longname="Some vertical profile", units="watermelons"),
        slice = (longname="Some slice", units="mushrooms")
    )

    global_attributes = (location="Bay of Fundy", onions=7)

    nc_filepath = "test_function_outputs_$(typeof(arch)).nc"

    simulation.output_writers[:food] =
        NetCDFOutputWriter(model, outputs; filepath=nc_filepath,
                           schedule=TimeInterval(Δt), dimensions=dims, array_type=Array{Float64}, verbose=true,
                           global_attributes=global_attributes, output_attributes=output_attributes)

    run!(simulation)

    ds = Dataset(nc_filepath, "r")

    @test haskey(ds.attrib, "date") && !isnothing(ds.attrib["date"])
    @test haskey(ds.attrib, "Julia") && !isnothing(ds.attrib["Julia"])
    @test haskey(ds.attrib, "Oceananigans") && !isnothing(ds.attrib["Oceananigans"])
    @test haskey(ds.attrib, "schedule") && !isnothing(ds.attrib["schedule"])
    @test haskey(ds.attrib, "interval") && !isnothing(ds.attrib["interval"])
    @test haskey(ds.attrib, "output time interval") && !isnothing(ds.attrib["output time interval"])

    @test eltype(ds["time"]) == eltype(model.clock.time)

    @test eltype(ds["xC"]) == Float64
    @test eltype(ds["xF"]) == Float64
    @test eltype(ds["yC"]) == Float64
    @test eltype(ds["yF"]) == Float64
    @test eltype(ds["zC"]) == Float64
    @test eltype(ds["zF"]) == Float64

    @test length(ds["xC"]) == N
    @test length(ds["yC"]) == N
    @test length(ds["zC"]) == N
    @test length(ds["xF"]) == N
    @test length(ds["yF"]) == N
    @test length(ds["zF"]) == N+1  # z is Bounded

    @test ds["xC"][1] == grid.xC[1]
    @test ds["xF"][1] == grid.xF[1]
    @test ds["yC"][1] == grid.yC[1]
    @test ds["yF"][1] == grid.yF[1]
    @test ds["zC"][1] == grid.zC[1]
    @test ds["zF"][1] == grid.zF[1]

    @test ds["xC"][end] == grid.xC[N]
    @test ds["yC"][end] == grid.yC[N]
    @test ds["zC"][end] == grid.zC[N]
    @test ds["xF"][end] == grid.xF[N]
    @test ds["yF"][end] == grid.yF[N]
    @test ds["zF"][end] == grid.zF[N+1]  # z is Bounded

    @test ds.attrib["location"] == "Bay of Fundy"
    @test ds.attrib["onions"] == 7

    @test eltype(ds["scalar"]) == Float64
    @test eltype(ds["profile"]) == Float64
    @test eltype(ds["slice"]) == Float64

    @test length(ds["time"]) == iters+1
    @test ds["time"][:] == [n*Δt for n in 0:iters]

    @test length(ds["scalar"]) == iters+1
    @test ds["scalar"].attrib["longname"] == "Some scalar"
    @test ds["scalar"].attrib["units"] == "bananas"
    @test ds["scalar"][:] == [(n*Δt)^2 for n in 0:iters]
    @test dimnames(ds["scalar"]) == ("time",)

    @test ds["profile"].attrib["longname"] == "Some vertical profile"
    @test ds["profile"].attrib["units"] == "watermelons"
    @test size(ds["profile"]) == (N, iters+1)
    @test dimnames(ds["profile"]) == ("zC", "time")

    for n in 0:iters
        @test ds["profile"][:, n+1] == n*Δt .* exp.(znodes(Center, grid))
    end

    @test ds["slice"].attrib["longname"] == "Some slice"
    @test ds["slice"].attrib["units"] == "mushrooms"
    @test size(ds["slice"]) == (N, N, iters+1)
    @test dimnames(ds["slice"]) == ("xC", "yC", "time")

    for n in 0:iters
        @test ds["slice"][:, :, n+1] == n*Δt .* (   sin.(xnodes(Center, grid, reshape=true)[:, :, 1])
                                                 .* cos.(ynodes(Face, grid, reshape=true)[:, :, 1]))
    end

    close(ds)

    #####
    ##### Take 1 more time step and test that appending to a NetCDF file works
    #####

    iters += 1
    simulation = Simulation(model, Δt=Δt, stop_iteration=iters)

    simulation.output_writers[:food] =
        NetCDFOutputWriter(model, outputs; filepath=nc_filepath, mode="a",
                           schedule=IterationInterval(1), array_type=Array{Float64}, dimensions=dims, verbose=true,
                           global_attributes=global_attributes, output_attributes=output_attributes)

    run!(simulation)

    ds = Dataset(nc_filepath, "r")

    @test length(ds["time"]) == iters+1
    @test length(ds["scalar"]) == iters+1
    @test size(ds["profile"]) == (N, iters+1)
    @test size(ds["slice"]) == (N, N, iters+1)

    @test ds["time"][:] == [n*Δt for n in 0:iters]
    @test ds["scalar"][:] == [(n*Δt)^2 for n in 0:iters]

    for n in 0:iters
        @test ds["profile"][:, n+1] == n*Δt .* exp.(znodes(Center, grid))
        @test ds["slice"][:, :, n+1] == n*Δt .* (   sin.(xnodes(Center, grid, reshape=true)[:, :, 1])
                                                 .* cos.(ynodes(Face, grid, reshape=true)[:, :, 1]))
    end

    close(ds)

    rm(nc_filepath)

    return nothing
end

#####
##### JLD2OutputWriter tests
#####

function jld2_field_output(model)

    model.clock.iteration = 0
    model.clock.time = 0.0

    set!(model, u = (x, y, z) -> rand(),
                v = (x, y, z) -> rand(),
                w = (x, y, z) -> rand(),
                T = 0,
                S = 0)

    simulation = Simulation(model, Δt=1.0, stop_iteration=1)

    simulation.output_writers[:velocities] = JLD2OutputWriter(model, model.velocities,
                                                                    schedule = TimeInterval(1),
                                                                        dir = ".",
                                                                     prefix = "test",
                                                                      force = true)

    u₀ = data(model.velocities.u)[3, 3, 3]
    v₀ = data(model.velocities.v)[3, 3, 3]
    w₀ = data(model.velocities.w)[3, 3, 3]

    run!(simulation)

    file = jldopen("test.jld2")

    # Data is saved without halos by default
    u₁ = file["timeseries/u/0"][3, 3, 3]
    v₁ = file["timeseries/v/0"][3, 3, 3]
    w₁ = file["timeseries/w/0"][3, 3, 3]

    close(file)

    rm("test.jld2")

    FT = typeof(u₁)

    return FT(u₀) == u₁ && FT(v₀) == v₁ && FT(w₀) == w₁
end

function jld2_sliced_field_output(model)

    model.clock.iteration = 0
    model.clock.time = 0.0

    set!(model, u = (x, y, z) -> rand(),
                v = (x, y, z) -> rand(),
                w = (x, y, z) -> rand())

    simulation = Simulation(model, Δt=1.0, stop_iteration=1)

    simulation.output_writers[:velocities] =
        JLD2OutputWriter(model, model.velocities,
                                      schedule = TimeInterval(1),
                                 field_slicer = FieldSlicer(i=1:2, j=1:3, k=:),
                                          dir = ".",
                                       prefix = "test",
                                        force = true)

    run!(simulation)

    file = jldopen("test.jld2")

    u₁ = file["timeseries/u/0"]
    v₁ = file["timeseries/v/0"]
    w₁ = file["timeseries/w/0"]

    close(file)

    rm("test.jld2")

    return size(u₁) == (2, 3, 4) && size(v₁) == (2, 3, 4) && size(w₁) == (2, 3, 5)
end

function run_jld2_file_splitting_tests(arch)
    model = IncompressibleModel(architecture=arch, grid=RegularRectilinearGrid(size=(16, 16, 16), extent=(1, 1, 1)))
    simulation = Simulation(model, Δt=1, stop_iteration=10)

    function fake_bc_init(file, model)
        file["boundary_conditions/fake"] = π
    end

    ow = JLD2OutputWriter(model, (u=model.velocities.u,); dir=".", prefix="test", schedule=IterationInterval(1),
                          init=fake_bc_init, including=[:grid],
                          field_slicer=nothing, array_type=Array{Float64},
                          max_filesize=200KiB, force=true)

    push!(simulation.output_writers, ow)

    # 531 KiB of output will be written which should get split into 3 files.
    run!(simulation)

    # Test that files has been split according to size as expected.
    @test filesize("test_part1.jld2") > 200KiB
    @test filesize("test_part2.jld2") > 200KiB
    @test filesize("test_part3.jld2") < 200KiB

    for n in string.(1:3)
        filename = "test_part$n.jld2"
        jldopen(filename, "r") do file
            # Test to make sure all files contain structs from `including`.
            @test file["grid/Nx"] == 16

            # Test to make sure all files contain info from `init` function.
            @test file["boundary_conditions/fake"] == π
        end

        # Leave test directory clean.
        rm(filename)
    end
end

#####
##### Checkpointer tests
#####

function test_model_equality(test_model, true_model)
    CUDA.@allowscalar begin
        @test all(test_model.velocities.u.data     .≈ true_model.velocities.u.data)
        @test all(test_model.velocities.v.data     .≈ true_model.velocities.v.data)
        @test all(test_model.velocities.w.data     .≈ true_model.velocities.w.data)
        @test all(test_model.tracers.T.data        .≈ true_model.tracers.T.data)
        @test all(test_model.tracers.S.data        .≈ true_model.tracers.S.data)
        @test all(test_model.timestepper.Gⁿ.u.data .≈ true_model.timestepper.Gⁿ.u.data)
        @test all(test_model.timestepper.Gⁿ.v.data .≈ true_model.timestepper.Gⁿ.v.data)
        @test all(test_model.timestepper.Gⁿ.w.data .≈ true_model.timestepper.Gⁿ.w.data)
        @test all(test_model.timestepper.Gⁿ.T.data .≈ true_model.timestepper.Gⁿ.T.data)
        @test all(test_model.timestepper.Gⁿ.S.data .≈ true_model.timestepper.Gⁿ.S.data)
        @test all(test_model.timestepper.G⁻.u.data .≈ true_model.timestepper.G⁻.u.data)
        @test all(test_model.timestepper.G⁻.v.data .≈ true_model.timestepper.G⁻.v.data)
        @test all(test_model.timestepper.G⁻.w.data .≈ true_model.timestepper.G⁻.w.data)
        @test all(test_model.timestepper.G⁻.T.data .≈ true_model.timestepper.G⁻.T.data)
        @test all(test_model.timestepper.G⁻.S.data .≈ true_model.timestepper.G⁻.S.data)
    end
    return nothing
end

"""
Run two coarse rising thermal bubble simulations and make sure

1. When restarting from a checkpoint, the restarted moded matches the non-restarted
   model to machine precision.

2. When using set!(new_model) to a checkpoint, the new model matches the non-restarted
   simulation to machine precision.

3. run!(new_model, pickup) works as expected
"""
function run_thermal_bubble_checkpointer_tests(arch)
    Nx, Ny, Nz = 16, 16, 16
    Lx, Ly, Lz = 100, 100, 100
    Δt = 6

    grid = RegularRectilinearGrid(size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz))
    closure = IsotropicDiffusivity(ν=4e-2, κ=4e-2)
    true_model = IncompressibleModel(architecture=arch, grid=grid, closure=closure)

    # Add a cube-shaped warm temperature anomaly that takes up the middle 50%
    # of the domain volume.
    i1, i2 = round(Int, Nx/4), round(Int, 3Nx/4)
    j1, j2 = round(Int, Ny/4), round(Int, 3Ny/4)
    k1, k2 = round(Int, Nz/4), round(Int, 3Nz/4)
    CUDA.@allowscalar true_model.tracers.T.data[i1:i2, j1:j2, k1:k2] .+= 0.01

    checkpointed_model = deepcopy(true_model)

    true_simulation = Simulation(true_model, Δt=Δt, stop_iteration=9)
    run!(true_simulation) # for 9 iterations

    checkpointed_simulation = Simulation(checkpointed_model, Δt=Δt, stop_iteration=5)
    checkpointer = Checkpointer(checkpointed_model, schedule=IterationInterval(5), force=true)
    push!(checkpointed_simulation.output_writers, checkpointer)

    # Checkpoint should be saved as "checkpoint5.jld" after the 5th iteration.
    run!(checkpointed_simulation) # for 5 iterations

    # model_kwargs = Dict{Symbol, Any}(:boundary_conditions => SolutionBoundaryConditions(grid))
    restored_model = restore_from_checkpoint("checkpoint_iteration5.jld2")

    #restored_simulation = Simulation(restored_model, Δt=Δt, stop_iteration=9)
    #run!(restored_simulation)

    for n in 1:4
        update_state!(restored_model)
        time_step!(restored_model, Δt, euler=false) # time-step for 4 iterations
    end

    test_model_equality(restored_model, true_model)

    #####
    ##### Test `set!(model, checkpoint_file)`
    #####

    new_model = IncompressibleModel(architecture=arch, grid=grid, closure=closure)

    set!(new_model, "checkpoint_iteration5.jld2")

    @test new_model.clock.iteration == checkpointed_model.clock.iteration
    @test new_model.clock.time == checkpointed_model.clock.time
    test_model_equality(new_model, checkpointed_model)

    #####
    ##### Test `run!(sim, pickup=true)
    #####

    new_simulation = Simulation(new_model, Δt=Δt, stop_iteration=9)

    # Pickup from explicit checkpoint path
    run!(new_simulation, pickup="checkpoint_iteration0.jld2")
    test_model_equality(new_model, true_model)

    run!(new_simulation, pickup="checkpoint_iteration5.jld2")
    test_model_equality(new_model, true_model)

    # Pickup using existing checkpointer
    new_simulation.output_writers[:checkpointer] =
        Checkpointer(new_model, schedule=IterationInterval(5), force=true)

    run!(new_simulation, pickup=true)
    test_model_equality(new_model, true_model)

    run!(new_simulation, pickup=0)
    test_model_equality(new_model, true_model)

    run!(new_simulation, pickup=5)
    test_model_equality(new_model, true_model)

    rm("checkpoint_iteration0.jld2", force=true)
    rm("checkpoint_iteration5.jld2", force=true)

    return nothing
end

function run_checkpoint_with_function_bcs_tests(arch)
    grid = RegularRectilinearGrid(size=(16, 16, 16), extent=(1, 1, 1))

    @inline some_flux(x, y, t) = 2x + exp(y)
    top_u_bc = top_T_bc = FluxBoundaryCondition(some_flux)
    u_bcs = UVelocityBoundaryConditions(grid, top=top_u_bc)
    T_bcs = TracerBoundaryConditions(grid, top=top_T_bc)

    model = IncompressibleModel(architecture=arch, grid=grid, boundary_conditions=(u=u_bcs, T=T_bcs))
    set!(model, u=π/2, v=ℯ, T=Base.MathConstants.γ, S=Base.MathConstants.φ)

    checkpointer = Checkpointer(model, schedule=IterationInterval(1))
    write_output!(checkpointer, model)
    model = nothing

    restored_model = restore_from_checkpoint("checkpoint_iteration0.jld2")
    @test  ismissing(restored_model.velocities.u.boundary_conditions)
    @test !ismissing(restored_model.velocities.v.boundary_conditions)
    @test !ismissing(restored_model.velocities.w.boundary_conditions)
    @test  ismissing(restored_model.tracers.T.boundary_conditions)
    @test !ismissing(restored_model.tracers.S.boundary_conditions)

    CUDA.@allowscalar begin
        @test all(interior(restored_model.velocities.u) .≈ π/2)
        @test all(interior(restored_model.velocities.v) .≈ ℯ)
        @test all(interior(restored_model.velocities.w) .== 0)
        @test all(interior(restored_model.tracers.T) .≈ Base.MathConstants.γ)
        @test all(interior(restored_model.tracers.S) .≈ Base.MathConstants.φ)
    end
    restored_model = nothing

    properly_restored_model = restore_from_checkpoint("checkpoint_iteration0.jld2",
                                                      boundary_conditions=(u=u_bcs, T=T_bcs))

    CUDA.@allowscalar begin
        @test all(interior(properly_restored_model.velocities.u) .≈ π/2)
        @test all(interior(properly_restored_model.velocities.v) .≈ ℯ)
        @test all(interior(properly_restored_model.velocities.w) .== 0)
        @test all(interior(properly_restored_model.tracers.T) .≈ Base.MathConstants.γ)
        @test all(interior(properly_restored_model.tracers.S) .≈ Base.MathConstants.φ)
    end

    @test !ismissing(properly_restored_model.velocities.u.boundary_conditions)
    @test !ismissing(properly_restored_model.velocities.v.boundary_conditions)
    @test !ismissing(properly_restored_model.velocities.w.boundary_conditions)
    @test !ismissing(properly_restored_model.tracers.T.boundary_conditions)
    @test !ismissing(properly_restored_model.tracers.S.boundary_conditions)

    u, v, w = properly_restored_model.velocities
    T, S = properly_restored_model.tracers

    @test u.boundary_conditions.x.left  isa PBC
    @test u.boundary_conditions.x.right isa PBC
    @test u.boundary_conditions.y.left  isa PBC
    @test u.boundary_conditions.y.right isa PBC
    @test u.boundary_conditions.z.left  isa ZFBC
    @test u.boundary_conditions.z.right isa FBC
    @test u.boundary_conditions.z.right.condition isa ContinuousBoundaryFunction
    @test u.boundary_conditions.z.right.condition.func(1, 2, 3) == some_flux(1, 2, 3)

    @test T.boundary_conditions.x.left  isa PBC
    @test T.boundary_conditions.x.right isa PBC
    @test T.boundary_conditions.y.left  isa PBC
    @test T.boundary_conditions.y.right isa PBC
    @test T.boundary_conditions.z.left  isa ZFBC
    @test T.boundary_conditions.z.right isa FBC
    @test T.boundary_conditions.z.right.condition isa ContinuousBoundaryFunction
    @test T.boundary_conditions.z.right.condition.func(1, 2, 3) == some_flux(1, 2, 3)

    # Test that the restored model can be time stepped
    time_step!(properly_restored_model, 1)
    @test properly_restored_model isa IncompressibleModel

    return nothing
end

function run_cross_architecture_checkpointer_tests(arch1, arch2)
    grid = RegularRectilinearGrid(size=(16, 16, 16), extent=(1, 1, 1))
    model = IncompressibleModel(architecture=arch1, grid=grid)
    set!(model, u=π/2, v=ℯ, T=Base.MathConstants.γ, S=Base.MathConstants.φ)

    checkpointer = Checkpointer(model, schedule=IterationInterval(1))
    write_output!(checkpointer, model)
    model = nothing

    restored_model = restore_from_checkpoint("checkpoint_iteration0.jld2", architecture=arch2)

    @test restored_model.architecture == arch2

    ArrayType = array_type(restored_model.architecture)
    CUDA.@allowscalar begin
        @test restored_model.velocities.u.data.parent isa ArrayType
        @test restored_model.velocities.v.data.parent isa ArrayType
        @test restored_model.velocities.w.data.parent isa ArrayType
        @test restored_model.tracers.T.data.parent isa ArrayType
        @test restored_model.tracers.S.data.parent isa ArrayType

        @test all(interior(restored_model.velocities.u) .≈ π/2)
        @test all(interior(restored_model.velocities.v) .≈ ℯ)
        @test all(interior(restored_model.velocities.w) .== 0)
        @test all(interior(restored_model.tracers.T) .≈ Base.MathConstants.γ)
        @test all(interior(restored_model.tracers.S) .≈ Base.MathConstants.φ)
    end

    # Test that the restored model can be time stepped
    time_step!(restored_model, 1)
    @test restored_model isa IncompressibleModel

    return nothing
end

function run_checkpointer_cleanup_tests(arch)
    grid = RegularRectilinearGrid(size=(1, 1, 1), extent=(1, 1, 1))
    model = IncompressibleModel(architecture=arch, grid=grid)
    simulation = Simulation(model, Δt=0.2, stop_iteration=10)

    simulation.output_writers[:checkpointer] = Checkpointer(model, schedule=IterationInterval(3), cleanup=true)
    run!(simulation)

    [@test !isfile("checkpoint_iteration$i.jld2") for i in 1:10 if i != 9]
    @test isfile("checkpoint_iteration9.jld2")

    return nothing
end

#####
##### Dependency adding tests
#####

function dependencies_added_correctly!(model, windowed_time_average, output_writer)

    model.clock.iteration = 0
    model.clock.time = 0.0

    simulation = Simulation(model, Δt=1.0, stop_iteration=1)
    push!(simulation.output_writers, output_writer)
    run!(simulation)

    return windowed_time_average ∈ values(simulation.diagnostics)
end

function run_dependency_adding_tests(model)

    windowed_time_average = WindowedTimeAverage(model.velocities.u, schedule=AveragedTimeInterval(4, window=2))

    output = Dict("time_average" => windowed_time_average)
    attributes = Dict("time_average" => Dict("longname" => "A time average",  "units" => "arbitrary"))
    dimensions = Dict("time_average" => ("xF", "yC", "zC"))

    # JLD2 dependencies test
    jld2_output_writer = JLD2OutputWriter(model, output, schedule=TimeInterval(4), dir=".", prefix="test", force=true)

    @test dependencies_added_correctly!(model, windowed_time_average, jld2_output_writer)

    # NetCDF dependency test
    netcdf_output_writer = NetCDFOutputWriter(model, output,
                                                       schedule = TimeInterval(4),
                                                       filepath = "test.nc",
                                              output_attributes = attributes,
                                                     dimensions = dimensions)

    @test dependencies_added_correctly!(model, windowed_time_average, netcdf_output_writer)

    rm("test.nc")

    return nothing
end

#####
##### Time averaging of output tests
#####

function run_windowed_time_averaging_simulation_tests!(model)
    model.clock.iteration = model.clock.time = 0
    simulation = Simulation(model, Δt=1.0, stop_iteration=0)

    jld2_output_writer = JLD2OutputWriter(model, model.velocities,
                                          schedule = AveragedTimeInterval(π, window=1),
                                            prefix = "test",
                                             force = true)

                                          # https://github.com/Alexander-Barth/NCDatasets.jl/issues/105
    nc_filepath1 = "windowed_time_average_test1.nc"
    nc_outputs = Dict(string(name) => field for (name, field) in pairs(model.velocities))
    nc_output_writer = NetCDFOutputWriter(model, nc_outputs, filepath=nc_filepath1,
                                          schedule = AveragedTimeInterval(π, window=1))

    jld2_outputs_are_time_averaged = Tuple(typeof(out) <: WindowedTimeAverage for out in jld2_output_writer.outputs)
      nc_outputs_are_time_averaged = Tuple(typeof(out) <: WindowedTimeAverage for out in values(nc_output_writer.outputs))

    @test all(jld2_outputs_are_time_averaged)
    @test all(nc_outputs_are_time_averaged)

    # Test that the collection does *not* start when a simulation is initialized
    # when time_interval ≠ time_averaging_window
    simulation.output_writers[:jld2] = jld2_output_writer
    simulation.output_writers[:nc] = nc_output_writer

    run!(simulation)

    jld2_u_windowed_time_average = simulation.output_writers[:jld2].outputs.u
    nc_w_windowed_time_average = simulation.output_writers[:nc].outputs["w"]

    @test !(jld2_u_windowed_time_average.schedule.collecting)
    @test !(nc_w_windowed_time_average.schedule.collecting)

    # Test that time-averaging is finalized prior to output even when averaging over
    # time_window is not fully realized. For this, step forward to a time at which
    # collection should start. Note that time_interval = π and time_window = 1.0.
    simulation.Δt = 1.5
    simulation.stop_iteration = 2
    run!(simulation) # model.clock.time = 3.0, just before output but after average-collection.

    @test jld2_u_windowed_time_average.schedule.collecting
    @test nc_w_windowed_time_average.schedule.collecting

    # Step forward such that time_window is not reached, but output will occur.
    simulation.Δt = π - 3 + 0.01 # ≈ 0.15 < 1.0
    simulation.stop_iteration = 3
    run!(simulation) # model.clock.time ≈ 3.15, after output

    @test jld2_u_windowed_time_average.schedule.previous_interval_stop_time ==
        model.clock.time - rem(model.clock.time, jld2_u_windowed_time_average.schedule.interval)

    @test nc_w_windowed_time_average.schedule.previous_interval_stop_time ==
        model.clock.time - rem(model.clock.time, nc_w_windowed_time_average.schedule.interval)

    # Test that collection does start when a simulation is initialized and
    # time_interval == time_averaging_window
    model.clock.iteration = model.clock.time = 0

    simulation.output_writers[:jld2] = JLD2OutputWriter(model, model.velocities,
                                                        schedule = AveragedTimeInterval(π, window=π),
                                                          prefix = "test",
                                                           force = true)

    nc_filepath2 = "windowed_time_average_test2.nc"
    nc_outputs = Dict(string(name) => field for (name, field) in pairs(model.velocities))
    simulation.output_writers[:nc] = NetCDFOutputWriter(model, nc_outputs, filepath=nc_filepath2,
                                                        schedule=AveragedTimeInterval(π, window=π))

    run!(simulation)

    @test simulation.output_writers[:jld2].outputs.u.schedule.collecting
    @test simulation.output_writers[:nc].outputs["w"].schedule.collecting

    rm(nc_filepath1)
    rm(nc_filepath2)

    return nothing
end

function jld2_time_averaging_of_horizontal_averages(model)

    model.clock.iteration = 0
    model.clock.time = 0.0

    set!(model, u = (x, y, z) -> 1,
                v = (x, y, z) -> 2,
                w = (x, y, z) -> 0,
                T = (x, y, z) -> 4)

    u, v, w = model.velocities
    T, S = model.tracers

    simulation = Simulation(model, Δt=1.0, stop_iteration=5)

    u, v, w = model.velocities
    T, S = model.tracers

    average_fluxes = (wu = AveragedField(w * u, dims=(1, 2)),
                      uv = AveragedField(u * v, dims=(1, 2)),
                      wT = AveragedField(w * T, dims=(1, 2)))

    simulation.output_writers[:fluxes] = JLD2OutputWriter(model, average_fluxes,
                                                          schedule = AveragedTimeInterval(4, window=2),
                                                               dir = ".",
                                                            prefix = "test",
                                                             force = true)

    run!(simulation)

    file = jldopen("test.jld2")

    # Data is saved without halos by default
    wu = file["timeseries/wu/4"][1, 1, 3]
    uv = file["timeseries/uv/4"][1, 1, 3]
    wT = file["timeseries/wT/4"][1, 1, 3]

    close(file)

    rm("test.jld2")

    FT = eltype(model.grid)

    return wu == zero(FT) && wT == zero(FT) && uv == FT(2)
end

function run_netcdf_time_averaging_tests(arch)
    topo = (Periodic, Periodic, Periodic)
    domain = (x=(0, 1), y=(0, 1), z=(0, 1))
    grid = RegularRectilinearGrid(topology=topo, size=(4, 4, 4); domain...)

    λ(x, y, z) = x + (1 - y)^2 + tanh(z)
    Fc(x, y, z, t, c) = - λ(x, y, z) * c

    c_forcing = Forcing(Fc, field_dependencies=(:c,))

    model = IncompressibleModel(
                grid = grid,
        architecture = arch,
         timestepper = :RungeKutta3,
             tracers = :c,
             forcing = (c=c_forcing,),
            coriolis = nothing,
            buoyancy = nothing,
             closure = nothing
    )

    set!(model, c=1)

    Δt = 1/512  # Nice floating-point number
    simulation = Simulation(model, Δt=Δt, stop_time=50Δt)

    ∫c_dxdy = AveragedField(model.tracers.c, dims=(1, 2))

    nc_outputs = Dict("c" => ∫c_dxdy)
    nc_dimensions = Dict("c" => ("zC",))

    horizontal_average_nc_filepath = "decay_averaged_field_test.nc"
    simulation.output_writers[:horizontal_average] =
        NetCDFOutputWriter(model, nc_outputs, filepath=horizontal_average_nc_filepath, schedule=TimeInterval(10Δt),
                           dimensions=nc_dimensions, array_type=Array{Float64}, verbose=true)

    time_average_nc_filepath = "decay_windowed_time_average_test.nc"
    window = 6Δt
    stride = 2
    simulation.output_writers[:time_average] =
        NetCDFOutputWriter(model, nc_outputs, filepath=time_average_nc_filepath, array_type=Array{Float64},
                           schedule=AveragedTimeInterval(10Δt, window=window, stride=stride),
                           dimensions=nc_dimensions, verbose=true)

    run!(simulation)

    ##### Horizontal average should evaluate to
    #####
    #####     c̄(z, t) = ∫₀¹ ∫₀¹ exp{- λ(x, y, z) * t} dx dy
    #####             = 1 / (Nx*Ny) * Σᵢ₌₁ᴺˣ Σⱼ₌₁ᴺʸ exp{- λ(i, j, k) * t}
    #####
    ##### which we can compute analytically.

    ds = NCDataset(horizontal_average_nc_filepath)

    Nx, Ny, Nz = size(grid)
    xs, ys, zs = nodes(model.tracers.c)

    c̄(z, t) = 1 / (Nx * Ny) * sum(exp(-λ(x, y, z) * t) for x in xs for y in ys)

    for (n, t) in enumerate(ds["time"])
        @test ds["c"][:, n] ≈ c̄.(zs, t)
    end

    close(ds)

    #####
    ##### Test strided windowed time average against analytic solution
    #####

    ds = NCDataset(time_average_nc_filepath)

    attribute_names = ("schedule", "interval", "output time interval",
                       "time_averaging_window", "time averaging window",
                       "time_averaging_stride", "time averaging stride")

    for name in attribute_names
        @test haskey(ds.attrib, name) && !isnothing(ds.attrib[name])
    end

    c̄(ts) = 1/length(ts) * sum(c̄.(zs, t) for t in ts)

    window_size = Int(window/Δt)
    for (n, t) in enumerate(ds["time"][2:end])
        averaging_times = [t - n*Δt for n in 0:stride:window_size-1]
        @test ds["c"][:, n+1] ≈ c̄(averaging_times)
    end

    close(ds)

    rm(horizontal_average_nc_filepath)
    rm(time_average_nc_filepath)

    return nothing
end

function run_netcdf_output_alignment_tests(arch)
    grid = RegularRectilinearGrid(size=(1, 1, 1), extent=(1, 1, 1))
    model = IncompressibleModel(architecture=arch, grid=grid)
    simulation = Simulation(model, Δt=0.2, stop_time=40)

    test_filename1 = "test_output_alignment1.nc"
    simulation.output_writers[:stuff] =
        NetCDFOutputWriter(model, model.velocities, filepath=test_filename1,
                           schedule=TimeInterval(7.3))

    test_filename2 = "test_output_alignment2.nc"
    simulation.output_writers[:something] =
        NetCDFOutputWriter(model, model.tracers, filepath=test_filename2,
                           schedule=TimeInterval(3.0))

    run!(simulation)

    Dataset(test_filename1, "r") do ds
        @test all(ds["time"] .== 0:7.3:40)
    end

    Dataset(test_filename2, "r") do ds
        @test all(ds["time"] .== 0:3.0:40)
    end

    rm(test_filename1)
    rm(test_filename2)

    return nothing
end

#####
##### Run output writer tests!
#####

@testset "Output writers" begin
    @info "Testing output writers..."

    @testset "FieldSlicer" begin
        @info "  Testing FieldSlicer..."
        @test FieldSlicer() isa FieldSlicer
    end

    for arch in archs
        # Some tests can reuse this same grid and model.
        grid = RegularRectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1),
                                    topology=(Periodic, Periodic, Bounded))

        model = IncompressibleModel(architecture=arch, grid=grid)

        @testset "WindowedTimeAverage" begin
            @info "  Testing WindowedTimeAverage..."
            @test instantiate_windowed_time_average(model)
            @test time_step_with_windowed_time_average(model)
        end

        @testset "NetCDF [$(typeof(arch))]" begin
            @info "  Testing NetCDF output writer [$(typeof(arch))]..."
            run_DateTime_netcdf_tests(arch)
            run_TimeDate_netcdf_tests(arch)
            run_thermal_bubble_netcdf_tests(arch)
            run_thermal_bubble_netcdf_tests_with_halos(arch)
            run_netcdf_function_output_tests(arch)
            run_netcdf_output_alignment_tests(arch)
        end

        @testset "JLD2 [$(typeof(arch))]" begin
            @info "  Testing JLD2 output writer [$(typeof(arch))]..."

            @test jld2_field_output(model)
            @test jld2_sliced_field_output(model)

            run_jld2_file_splitting_tests(arch)
        end

        @testset "Checkpointer [$(typeof(arch))]" begin
            @info "  Testing Checkpointer [$(typeof(arch))]..."

            run_thermal_bubble_checkpointer_tests(arch)
            run_checkpoint_with_function_bcs_tests(arch)

            @hascuda run_cross_architecture_checkpointer_tests(CPU(), GPU())
            @hascuda run_cross_architecture_checkpointer_tests(GPU(), CPU())

            run_checkpointer_cleanup_tests(arch)
        end

        @testset "Dependency adding [$(typeof(arch))]" begin
            @info "    Testing dependency adding [$(typeof(arch))]..."
            run_dependency_adding_tests(model)
        end

        @testset "Time averaging of output [$(typeof(arch))]" begin
            @info "    Testing time averaging of output [$(typeof(arch))]..."

            run_windowed_time_averaging_simulation_tests!(model)

            @test jld2_time_averaging_of_horizontal_averages(model)

            run_netcdf_time_averaging_tests(arch)
        end
    end
end
