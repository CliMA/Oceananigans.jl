include("dependencies_for_runtests.jl")

using TimesDates: TimeDate
using Dates: DateTime, Nanosecond, Millisecond
using TimesDates: TimeDate

using CUDA
using NCDatasets

using Oceananigans: Clock
using Oceananigans.Models.HydrostaticFreeSurfaceModels: VectorInvariant

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

    Arch = typeof(arch)
    filepath = "test_datetime_$Arch.nc"
    isfile(filepath) && rm(filepath)

    simulation.output_writers[:netcdf] =
        NetCDFOutputWriter(model, fields(model);
            filename = filepath,
            schedule = IterationInterval(1),
            include_grid_metrics = false
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

    Arch = typeof(arch)
    filepath = "test_timedate_$Arch.nc"
    isfile(filepath) && rm(filepath)

    simulation.output_writers[:netcdf] =
        NetCDFOutputWriter(model, fields(model);
            filename = filepath,
            schedule = IterationInterval(1),
            include_grid_metrics = false
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

function test_netcdf_grid_metrics_rectilinear(arch, FT)
    Nx, Ny, Nz = 8, 6, 4
    Hx, Hy, Hz = 1, 2, 3

    grid = RectilinearGrid(arch,
        topology = (Periodic, Bounded, Bounded),
        size = (Nx, Ny, Nz),
        halo = (Hx, Hy, Hz),
        extent = (1, 2, 3)
    )

    model = NonhydrostaticModel(; grid,
        closure = ScalarDiffusivity(ν=4e-2, κ=4e-2),
        buoyancy = SeawaterBuoyancy(),
        tracers = (:T, :S)
    )

    Nt = 10
    simulation = Simulation(model, Δt=0.1, stop_iteration=Nt)

    Arch = typeof(arch)
    filepath_metrics_halos = "test_grid_metrics_rectilinear_halos_$(Arch)_$FT.nc"
    isfile(filepath_metrics_halos) && rm(filepath_metrics_halos)

    simulation.output_writers[:with_metrics_and_halos] =
        NetCDFOutputWriter(model, fields(model),
            filename = filepath_metrics_halos,
            schedule = IterationInterval(1),
            array_type = Array{FT},
            with_halos = true,
            include_grid_metrics = true,
            verbose = true
        )

    filepath_metrics_nohalos = "test_grid_metrics_rectilinear_nohalos_$(Arch)_$FT.nc"
    isfile(filepath_metrics_nohalos) && rm(filepath_metrics_nohalos)

    simulation.output_writers[:with_metrics_no_halos] =
        NetCDFOutputWriter(model, fields(model),
            filename = filepath_metrics_nohalos,
            schedule = IterationInterval(1),
            array_type = Array{FT},
            with_halos = false,
            include_grid_metrics = true,
            verbose = true
        )

    filepath_nometrics = "test_grid_metrics_rectilinear_nometrics_$(Arch)_$FT.nc"
    isfile(filepath_nometrics) && rm(filepath_nometrics)

    simulation.output_writers[:no_metrics] =
        NetCDFOutputWriter(model, fields(model),
            filename = filepath_nometrics,
            schedule = IterationInterval(1),
            array_type = Array{FT},
            with_halos = true,
            include_grid_metrics = false,
            verbose = true
        )

    i_slice = Colon()
    j_slice = 2:4
    k_slice = Nz

    nx = Nx
    ny = length(j_slice)
    nz = 1

    filepath_sliced = "test_grid_metrics_rectilinear_sliced_$(Arch)_$FT.nc"
    isfile(filepath_sliced) && rm(filepath_sliced)

    simulation.output_writers[:sliced] =
        NetCDFOutputWriter(model, fields(model),
            filename = filepath_sliced,
            indices = (i_slice, j_slice, k_slice),
            schedule = IterationInterval(1),
            array_type = Array{FT},
            with_halos = false,
            include_grid_metrics = true,
            verbose = true
        )

    run!(simulation)

    # Test NetCDF output with metrics and halos
    ds_mh = NCDataset(filepath_metrics_halos)

    @test haskey(ds_mh, "time")
    @test eltype(ds_mh["time"]) == Float64

    dims = ("x_f", "x_c", "y_f", "y_c", "z_f", "z_c")
    metrics = ("dx_f", "dx_c", "dy_f", "dy_c", "dz_f", "dz_c")
    vars = ("u", "v", "w", "T", "S")

    for var in (dims..., metrics..., vars...)
        @test haskey(ds_mh, var)
        @test haskey(ds_mh[var].attrib, "long_name")
        @test haskey(ds_mh[var].attrib, "units")
        @test eltype(ds_mh[var]) == FT
    end

    @test dimsize(ds_mh["time"]) == (time=Nt + 1,)

    @test dimsize(ds_mh[:x_f]) == (x_f=Nx + 2Hx,)
    @test dimsize(ds_mh[:x_c]) == (x_c=Nx + 2Hx,)
    @test dimsize(ds_mh[:y_f]) == (y_f=Ny + 2Hy + 1,)
    @test dimsize(ds_mh[:y_c]) == (y_c=Ny + 2Hy,)
    @test dimsize(ds_mh[:z_f]) == (z_f=Nz + 2Hz + 1,)
    @test dimsize(ds_mh[:z_c]) == (z_c=Nz + 2Hz,)

    @test dimsize(ds_mh[:dx_f]) == (x_f=Nx + 2Hx,)
    @test dimsize(ds_mh[:dx_c]) == (x_c=Nx + 2Hx,)
    @test dimsize(ds_mh[:dy_f]) == (y_f=Ny + 2Hy + 1,)
    @test dimsize(ds_mh[:dy_c]) == (y_c=Ny + 2Hy,)
    @test dimsize(ds_mh[:dz_f]) == (z_f=Nz + 2Hz + 1,)
    @test dimsize(ds_mh[:dz_c]) == (z_c=Nz + 2Hz,)

    @test dimsize(ds_mh[:u]) == (x_f=Nx + 2Hx, y_c=Ny + 2Hy,     z_c=Nz + 2Hz,     time=Nt + 1)
    @test dimsize(ds_mh[:v]) == (x_c=Nx + 2Hx, y_f=Ny + 2Hy + 1, z_c=Nz + 2Hz,     time=Nt + 1)
    @test dimsize(ds_mh[:w]) == (x_c=Nx + 2Hx, y_c=Ny + 2Hy,     z_f=Nz + 2Hz + 1, time=Nt + 1)
    @test dimsize(ds_mh[:T]) == (x_c=Nx + 2Hx, y_c=Ny + 2Hy,     z_c=Nz + 2Hz,     time=Nt + 1)
    @test dimsize(ds_mh[:S]) == (x_c=Nx + 2Hx, y_c=Ny + 2Hy,     z_c=Nz + 2Hz,     time=Nt + 1)

    close(ds_mh)
    rm(filepath_metrics_halos)

    # Test NetCDF output with metrics but not halos
    ds_m = NCDataset(filepath_metrics_nohalos)

    @test haskey(ds_m, "time")
    @test eltype(ds_m["time"]) == Float64

    for var in (dims..., metrics..., vars...)
        @test haskey(ds_m, var)
        @test haskey(ds_m[var].attrib, "long_name")
        @test haskey(ds_m[var].attrib, "units")
        @test eltype(ds_m[var]) == FT
    end

    @test dimsize(ds_m[:x_f]) == (x_f=Nx,)
    @test dimsize(ds_m[:x_c]) == (x_c=Nx,)
    @test dimsize(ds_m[:y_f]) == (y_f=Ny + 1,)
    @test dimsize(ds_m[:y_c]) == (y_c=Ny,)
    @test dimsize(ds_m[:z_f]) == (z_f=Nz + 1,)
    @test dimsize(ds_m[:z_c]) == (z_c=Nz,)

    @test dimsize(ds_m[:dx_f]) == (x_f=Nx,)
    @test dimsize(ds_m[:dx_c]) == (x_c=Nx,)
    @test dimsize(ds_m[:dy_f]) == (y_f=Ny + 1,)
    @test dimsize(ds_m[:dy_c]) == (y_c=Ny,)
    @test dimsize(ds_m[:dz_f]) == (z_f=Nz + 1,)
    @test dimsize(ds_m[:dz_c]) == (z_c=Nz,)

    @test dimsize(ds_m[:u]) == (x_f=Nx, y_c=Ny,     z_c=Nz,     time=Nt + 1)
    @test dimsize(ds_m[:v]) == (x_c=Nx, y_f=Ny + 1, z_c=Nz,     time=Nt + 1)
    @test dimsize(ds_m[:w]) == (x_c=Nx, y_c=Ny,     z_f=Nz + 1, time=Nt + 1)
    @test dimsize(ds_m[:T]) == (x_c=Nx, y_c=Ny,     z_c=Nz,     time=Nt + 1)
    @test dimsize(ds_m[:S]) == (x_c=Nx, y_c=Ny,     z_c=Nz,     time=Nt + 1)

    close(ds_m)
    rm(filepath_metrics_nohalos)

    # Test NetCDF output with no metrics (with halos)
    ds_h = NCDataset(filepath_nometrics)

    @test haskey(ds_h, "time")
    @test eltype(ds_h["time"]) == Float64

    for var in (dims..., vars...)
        @test haskey(ds_h, var)
        @test haskey(ds_h[var].attrib, "long_name")
        @test haskey(ds_h[var].attrib, "units")
        @test eltype(ds_h[var]) == FT
    end

    for metric in metrics
        @test !haskey(ds_h, metric)
    end

    @test dimsize(ds_h["time"]) == (time=Nt + 1,)

    @test dimsize(ds_h[:x_f]) == (x_f=Nx + 2Hx,)
    @test dimsize(ds_h[:x_c]) == (x_c=Nx + 2Hx,)
    @test dimsize(ds_h[:y_f]) == (y_f=Ny + 2Hy + 1,)
    @test dimsize(ds_h[:y_c]) == (y_c=Ny + 2Hy,)
    @test dimsize(ds_h[:z_f]) == (z_f=Nz + 2Hz + 1,)
    @test dimsize(ds_h[:z_c]) == (z_c=Nz + 2Hz,)

    @test dimsize(ds_h[:u]) == (x_f=Nx + 2Hx, y_c=Ny + 2Hy,     z_c=Nz + 2Hz,     time=Nt + 1)
    @test dimsize(ds_h[:v]) == (x_c=Nx + 2Hx, y_f=Ny + 2Hy + 1, z_c=Nz + 2Hz,     time=Nt + 1)
    @test dimsize(ds_h[:w]) == (x_c=Nx + 2Hx, y_c=Ny + 2Hy,     z_f=Nz + 2Hz + 1, time=Nt + 1)
    @test dimsize(ds_h[:T]) == (x_c=Nx + 2Hx, y_c=Ny + 2Hy,     z_c=Nz + 2Hz,     time=Nt + 1)
    @test dimsize(ds_h[:S]) == (x_c=Nx + 2Hx, y_c=Ny + 2Hy,     z_c=Nz + 2Hz,     time=Nt + 1)

    close(ds_h)
    rm(filepath_nometrics)

    # Test NetCDF sliced output with metrics
    ds_s = NCDataset(filepath_sliced)

    @test haskey(ds_s, "time")
    @test eltype(ds_s["time"]) == Float64

    for var in (dims..., metrics..., vars...)
        @test haskey(ds_s, var)
        @test haskey(ds_s[var].attrib, "long_name")
        @test haskey(ds_s[var].attrib, "units")
        @test eltype(ds_s[var]) == FT
    end

    @test dimsize(ds_s[:x_f]) == (x_f=nx,)
    @test dimsize(ds_s[:x_c]) == (x_c=nx,)
    @test dimsize(ds_s[:y_f]) == (y_f=ny,)
    @test dimsize(ds_s[:y_c]) == (y_c=ny,)
    @test dimsize(ds_s[:z_f]) == (z_f=nz,)
    @test dimsize(ds_s[:z_c]) == (z_c=nz,)

    @test dimsize(ds_s[:dx_f]) == (x_f=nx,)
    @test dimsize(ds_s[:dx_c]) == (x_c=nx,)
    @test dimsize(ds_s[:dy_f]) == (y_f=ny,)
    @test dimsize(ds_s[:dy_c]) == (y_c=ny,)
    @test dimsize(ds_s[:dz_f]) == (z_f=nz,)
    @test dimsize(ds_s[:dz_c]) == (z_c=nz,)

    @test dimsize(ds_s[:u]) == (x_f=nx, y_c=ny, z_c=nz, time=Nt + 1)
    @test dimsize(ds_s[:v]) == (x_c=nx, y_f=ny, z_c=nz, time=Nt + 1)
    @test dimsize(ds_s[:w]) == (x_c=nx, y_c=ny, z_f=nz, time=Nt + 1)
    @test dimsize(ds_s[:T]) == (x_c=nx, y_c=ny, z_c=nz, time=Nt + 1)
    @test dimsize(ds_s[:S]) == (x_c=nx, y_c=ny, z_c=nz, time=Nt + 1)

    close(ds_s)
    rm(filepath_sliced)

    return nothing
end

function test_netcdf_rectilinear_flat_xy(arch)
    Nx, Ny = 8, 8
    Hx, Hy = 2, 3

    grid = RectilinearGrid(arch,
        topology = (Periodic, Bounded, Flat),
        size = (Nx, Ny),
        halo = (Hx, Hy),
        extent = (π, 7)
    )

    model = NonhydrostaticModel(; grid,
        closure = ScalarDiffusivity(ν=4e-2, κ=4e-2),
        buoyancy = SeawaterBuoyancy(),
        tracers = (:T, :S)
    )

    Nt = 7
    simulation = Simulation(model, Δt=0.1, stop_iteration=Nt)

    Arch = typeof(arch)
    filepath_with_halos = "test_netcdf_rectilinear_flat_xy_$Arch.nc"
    isfile(filepath_with_halos) && rm(filepath_with_halos)

    simulation.output_writers[:with_halos] =
        NetCDFOutputWriter(model, fields(model),
            filename = filepath_with_halos,
            schedule = IterationInterval(1),
            array_type = Array{Float64},
            with_halos = true,
            include_grid_metrics = true,
            verbose = true
        )

    i_slice = 3:6
    j_slice = Ny

    nx = length(i_slice)
    ny = 1

    filepath_sliced = "test_netcdf_rectilinear_flat_xy_sliced_$(Arch).nc"
    isfile(filepath_sliced) && rm(filepath_sliced)

    simulation.output_writers[:sliced] =
        NetCDFOutputWriter(model, fields(model),
            filename = filepath_sliced,
            indices = (i_slice, j_slice, :),
            schedule = IterationInterval(1),
            array_type = Array{Float64},
            with_halos = false,
            include_grid_metrics = true,
            verbose = true
        )

    run!(simulation)

    # Test NetCDF output with halos
    ds_h = NCDataset(filepath_with_halos)

    dims = ("x_f", "x_c", "y_f", "y_c")
    not_dims = ("z_f", "z_c")

    metrics = ("dx_f", "dx_c", "dy_f", "dy_c")
    not_metrics = ("dz_f", "dz_c")

    vars = ("u", "v", "w", "T", "S")

    for var in (dims..., metrics..., vars...)
        @test haskey(ds_h, var)
        @test haskey(ds_h[var].attrib, "long_name")
        @test haskey(ds_h[var].attrib, "units")
        @test eltype(ds_h[var]) == Float64
    end

    for var in (not_dims..., not_metrics...)
        @test !haskey(ds_h, var)
    end

    @test dimsize(ds_h[:x_f]) == (x_f=Nx + 2Hx,)
    @test dimsize(ds_h[:x_c]) == (x_c=Nx + 2Hx,)
    @test dimsize(ds_h[:y_f]) == (y_f=Ny + 2Hy + 1,)
    @test dimsize(ds_h[:y_c]) == (y_c=Ny + 2Hy,)

    @test dimsize(ds_h[:dx_f]) == (x_f=Nx + 2Hx,)
    @test dimsize(ds_h[:dx_c]) == (x_c=Nx + 2Hx,)
    @test dimsize(ds_h[:dy_f]) == (y_f=Ny + 2Hy + 1,)
    @test dimsize(ds_h[:dy_c]) == (y_c=Ny + 2Hy,)

    @test dimsize(ds_h[:u]) == (x_f=Nx + 2Hx, y_c=Ny + 2Hy,     time=Nt + 1)
    @test dimsize(ds_h[:v]) == (x_c=Nx + 2Hx, y_f=Ny + 2Hy + 1, time=Nt + 1)
    @test dimsize(ds_h[:w]) == (x_c=Nx + 2Hx, y_c=Ny + 2Hy,     time=Nt + 1)
    @test dimsize(ds_h[:T]) == (x_c=Nx + 2Hx, y_c=Ny + 2Hy,     time=Nt + 1)
    @test dimsize(ds_h[:S]) == (x_c=Nx + 2Hx, y_c=Ny + 2Hy,     time=Nt + 1)

    close(ds_h)
    rm(filepath_with_halos)

    # Test NetCDF sliced output
    ds_s = NCDataset(filepath_sliced)

    for var in (dims..., metrics..., vars...)
        @test haskey(ds_s, var)
        @test haskey(ds_s[var].attrib, "long_name")
        @test haskey(ds_s[var].attrib, "units")
        @test eltype(ds_s[var]) == Float64
    end

    for var in (not_dims..., not_metrics...)
        @test !haskey(ds_s, var)
    end

    @test dimsize(ds_s[:x_f]) == (x_f=nx,)
    @test dimsize(ds_s[:x_c]) == (x_c=nx,)
    @test dimsize(ds_s[:y_f]) == (y_f=ny,)
    @test dimsize(ds_s[:y_c]) == (y_c=ny,)

    @test dimsize(ds_s[:dx_f]) == (x_f=nx,)
    @test dimsize(ds_s[:dx_c]) == (x_c=nx,)
    @test dimsize(ds_s[:dy_f]) == (y_f=ny,)
    @test dimsize(ds_s[:dy_c]) == (y_c=ny,)

    @test dimsize(ds_s[:u]) == (x_f=nx, y_c=ny, time=Nt + 1)
    @test dimsize(ds_s[:v]) == (x_c=nx, y_f=ny, time=Nt + 1)
    @test dimsize(ds_s[:w]) == (x_c=nx, y_c=ny, time=Nt + 1)
    @test dimsize(ds_s[:T]) == (x_c=nx, y_c=ny, time=Nt + 1)
    @test dimsize(ds_s[:S]) == (x_c=nx, y_c=ny, time=Nt + 1)

    close(ds_s)
    rm(filepath_sliced)

    return nothing
end

function test_netcdf_rectilinear_flat_xz(arch)
    Nx, Nz = 8, 8
    Hx, Hz = 2, 3

    grid = RectilinearGrid(arch,
        topology = (Periodic, Flat, Bounded),
        size = (Nx, Nz),
        halo = (Hx, Hz),
        extent = (π, 7)
    )

    model = NonhydrostaticModel(; grid,
        closure = ScalarDiffusivity(ν=4e-2, κ=4e-2),
        buoyancy = SeawaterBuoyancy(),
        tracers = (:T, :S)
    )

    Nt = 7
    simulation = Simulation(model, Δt=0.1, stop_iteration=Nt)

    Arch = typeof(arch)
    filepath_with_halos = "test_netcdf_rectilinear_flat_xz_$Arch.nc"
    isfile(filepath_with_halos) && rm(filepath_with_halos)

    simulation.output_writers[:with_halos] =
        NetCDFOutputWriter(model, fields(model),
            filename = filepath_with_halos,
            schedule = IterationInterval(1),
            array_type = Array{Float64},
            with_halos = true,
            include_grid_metrics = true,
            verbose = true
        )

    i_slice = 3:6
    k_slice = Nz

    nx = length(i_slice)
    nz = 1

    filepath_sliced = "test_netcdf_rectilinear_flat_xz_sliced_$(Arch).nc"
    isfile(filepath_sliced) && rm(filepath_sliced)

    simulation.output_writers[:sliced] =
        NetCDFOutputWriter(model, fields(model),
            filename = filepath_sliced,
            indices = (i_slice, :, k_slice),
            schedule = IterationInterval(1),
            array_type = Array{Float64},
            with_halos = false,
            include_grid_metrics = true,
            verbose = true
        )

    run!(simulation)

    # Test NetCDF output with halos
    ds_h = NCDataset(filepath_with_halos)

    dims = ("x_f", "x_c", "z_f", "z_c")
    not_dims = ("y_f", "y_c")

    metrics = ("dx_f", "dx_c", "dz_f", "dz_c")
    not_metrics = ("dy_f", "dy_c")

    vars = ("u", "v", "w", "T", "S")

    for var in (dims..., metrics..., vars...)
        @test haskey(ds_h, var)
        @test haskey(ds_h[var].attrib, "long_name")
        @test haskey(ds_h[var].attrib, "units")
        @test eltype(ds_h[var]) == Float64
    end

    for var in (not_dims..., not_metrics...)
        @test !haskey(ds_h, var)
    end

    @test dimsize(ds_h[:x_f]) == (x_f=Nx + 2Hx,)
    @test dimsize(ds_h[:x_c]) == (x_c=Nx + 2Hx,)
    @test dimsize(ds_h[:z_f]) == (z_f=Nz + 2Hz + 1,)
    @test dimsize(ds_h[:z_c]) == (z_c=Nz + 2Hz,)

    @test dimsize(ds_h[:dx_f]) == (x_f=Nx + 2Hx,)
    @test dimsize(ds_h[:dx_c]) == (x_c=Nx + 2Hx,)
    @test dimsize(ds_h[:dz_f]) == (z_f=Nz + 2Hz + 1,)
    @test dimsize(ds_h[:dz_c]) == (z_c=Nz + 2Hz,)

    @test dimsize(ds_h[:u]) == (x_f=Nx + 2Hx, z_c=Nz + 2Hz,     time=Nt + 1)
    @test dimsize(ds_h[:v]) == (x_c=Nx + 2Hx, z_c=Nz + 2Hz,     time=Nt + 1)
    @test dimsize(ds_h[:w]) == (x_c=Nx + 2Hx, z_f=Nz + 2Hz + 1, time=Nt + 1)
    @test dimsize(ds_h[:T]) == (x_c=Nx + 2Hx, z_c=Nz + 2Hz,     time=Nt + 1)
    @test dimsize(ds_h[:S]) == (x_c=Nx + 2Hx, z_c=Nz + 2Hz,     time=Nt + 1)

    close(ds_h)
    rm(filepath_with_halos)

    # Test NetCDF sliced output
    ds_s = NCDataset(filepath_sliced)

    for var in (dims..., metrics..., vars...)
        @test haskey(ds_s, var)
        @test haskey(ds_s[var].attrib, "long_name")
        @test haskey(ds_s[var].attrib, "units")
        @test eltype(ds_s[var]) == Float64
    end

    for var in (not_dims..., not_metrics...)
        @test !haskey(ds_s, var)
    end

    @test dimsize(ds_s[:x_f]) == (x_f=nx,)
    @test dimsize(ds_s[:x_c]) == (x_c=nx,)
    @test dimsize(ds_s[:z_f]) == (z_f=nz,)
    @test dimsize(ds_s[:z_c]) == (z_c=nz,)

    @test dimsize(ds_s[:dx_f]) == (x_f=nx,)
    @test dimsize(ds_s[:dx_c]) == (x_c=nx,)
    @test dimsize(ds_s[:dz_f]) == (z_f=nz,)
    @test dimsize(ds_s[:dz_c]) == (z_c=nz,)

    @test dimsize(ds_s[:u]) == (x_f=nx, z_c=nz, time=Nt + 1)
    @test dimsize(ds_s[:v]) == (x_c=nx, z_c=nz, time=Nt + 1)
    @test dimsize(ds_s[:w]) == (x_c=nx, z_f=nz, time=Nt + 1)
    @test dimsize(ds_s[:T]) == (x_c=nx, z_c=nz, time=Nt + 1)
    @test dimsize(ds_s[:S]) == (x_c=nx, z_c=nz, time=Nt + 1)

    close(ds_s)
    rm(filepath_sliced)

    return nothing
end

function test_netcdf_rectilinear_flat_yz(arch)
    Ny, Nz = 8, 8
    Hy, Hz = 2, 3

    grid = RectilinearGrid(arch,
        topology = (Flat, Periodic, Bounded),
        size = (Ny, Nz),
        halo = (Hy, Hz),
        extent = (π, 7)
    )

    model = NonhydrostaticModel(; grid,
        closure = ScalarDiffusivity(ν=4e-2, κ=4e-2),
        buoyancy = SeawaterBuoyancy(),
        tracers = (:T, :S)
    )

    Nt = 7
    simulation = Simulation(model, Δt=0.1, stop_iteration=Nt)

    Arch = typeof(arch)
    filepath_with_halos = "test_netcdf_rectilinear_flat_yz_$Arch.nc"
    isfile(filepath_with_halos) && rm(filepath_with_halos)

    simulation.output_writers[:with_halos] =
        NetCDFOutputWriter(model, fields(model),
            filename = filepath_with_halos,
            schedule = IterationInterval(1),
            array_type = Array{Float64},
            with_halos = true,
            include_grid_metrics = true,
            verbose = true
        )

    j_slice = 3:6
    k_slice = Nz

    ny = length(j_slice)
    nz = 1

    filepath_sliced = "test_netcdf_rectilinear_flat_yz_sliced_$(Arch).nc"
    isfile(filepath_sliced) && rm(filepath_sliced)

    simulation.output_writers[:sliced] =
        NetCDFOutputWriter(model, fields(model),
            filename = filepath_sliced,
            indices = (:, j_slice, k_slice),
            schedule = IterationInterval(1),
            array_type = Array{Float64},
            with_halos = false,
            include_grid_metrics = true,
            verbose = true
        )

    run!(simulation)

    # Test NetCDF output with halos
    ds_h = NCDataset(filepath_with_halos)

    dims = ("y_f", "y_c", "z_f", "z_c")
    not_dims = ("x_f", "x_c")

    metrics = ("dy_f", "dy_c", "dz_f", "dz_c")
    not_metrics = ("dx_f", "dx_c")

    vars = ("u", "v", "w", "T", "S")

    for var in (dims..., metrics..., vars...)
        @test haskey(ds_h, var)
        @test haskey(ds_h[var].attrib, "long_name")
        @test haskey(ds_h[var].attrib, "units")
        @test eltype(ds_h[var]) == Float64
    end

    for var in (not_dims..., not_metrics...)
        @test !haskey(ds_h, var)
    end

    @test dimsize(ds_h[:y_f]) == (y_f=Ny + 2Hy,)
    @test dimsize(ds_h[:y_c]) == (y_c=Ny + 2Hy,)
    @test dimsize(ds_h[:z_f]) == (z_f=Nz + 2Hz + 1,)
    @test dimsize(ds_h[:z_c]) == (z_c=Nz + 2Hz,)

    @test dimsize(ds_h[:dy_f]) == (y_f=Ny + 2Hy,)
    @test dimsize(ds_h[:dy_c]) == (y_c=Ny + 2Hy,)
    @test dimsize(ds_h[:dz_f]) == (z_f=Nz + 2Hz + 1,)
    @test dimsize(ds_h[:dz_c]) == (z_c=Nz + 2Hz,)

    @test dimsize(ds_h[:u]) == (y_c=Ny + 2Hy, z_c=Nz + 2Hz,     time=Nt + 1)
    @test dimsize(ds_h[:v]) == (y_f=Ny + 2Hy, z_c=Nz + 2Hz,     time=Nt + 1)
    @test dimsize(ds_h[:w]) == (y_c=Ny + 2Hy, z_f=Nz + 2Hz + 1, time=Nt + 1)
    @test dimsize(ds_h[:T]) == (y_c=Ny + 2Hy, z_c=Nz + 2Hz,     time=Nt + 1)
    @test dimsize(ds_h[:S]) == (y_c=Ny + 2Hy, z_c=Nz + 2Hz,     time=Nt + 1)

    close(ds_h)
    rm(filepath_with_halos)

    # Test NetCDF sliced output
    ds_s = NCDataset(filepath_sliced)

    for var in (dims..., metrics..., vars...)
        @test haskey(ds_s, var)
        @test haskey(ds_s[var].attrib, "long_name")
        @test haskey(ds_s[var].attrib, "units")
        @test eltype(ds_s[var]) == Float64
    end

    for var in (not_dims..., not_metrics...)
        @test !haskey(ds_s, var)
    end

    @test dimsize(ds_s[:y_f]) == (y_f=ny,)
    @test dimsize(ds_s[:y_c]) == (y_c=ny,)
    @test dimsize(ds_s[:z_f]) == (z_f=nz,)
    @test dimsize(ds_s[:z_c]) == (z_c=nz,)

    @test dimsize(ds_s[:dy_f]) == (y_f=ny,)
    @test dimsize(ds_s[:dy_c]) == (y_c=ny,)
    @test dimsize(ds_s[:dz_f]) == (z_f=nz,)
    @test dimsize(ds_s[:dz_c]) == (z_c=nz,)

    @test dimsize(ds_s[:u]) == (y_c=ny, z_c=nz, time=Nt + 1)
    @test dimsize(ds_s[:v]) == (y_f=ny, z_c=nz, time=Nt + 1)
    @test dimsize(ds_s[:w]) == (y_c=ny, z_f=nz, time=Nt + 1)
    @test dimsize(ds_s[:T]) == (y_c=ny, z_c=nz, time=Nt + 1)
    @test dimsize(ds_s[:S]) == (y_c=ny, z_c=nz, time=Nt + 1)

    close(ds_s)
    rm(filepath_sliced)

    return nothing
end

function test_netcdf_rectilinear_column(arch)
    N = 17
    H = 2

    grid = RectilinearGrid(arch,
        topology = (Flat, Flat, Bounded),
        size = N,
        halo = H,
        z = (-55, 0)
    )

    model = NonhydrostaticModel(; grid,
        closure = ScalarDiffusivity(ν=4e-2, κ=4e-2),
        buoyancy = SeawaterBuoyancy(),
        tracers = (:T, :S)
    )

    Nt = 5
    simulation = Simulation(model, Δt=0.1, stop_iteration=Nt)

    Arch = typeof(arch)
    filepath_with_halos = "test_netcdf_rectilinear_column_$Arch.nc"
    isfile(filepath_with_halos) && rm(filepath_with_halos)

    simulation.output_writers[:with_halos] =
        NetCDFOutputWriter(model, fields(model),
            filename = filepath_with_halos,
            schedule = IterationInterval(1),
            array_type = Array{Float64},
            with_halos = true,
            include_grid_metrics = true,
            verbose = true
        )

    k_slice = 5:10
    n = length(k_slice)

    filepath_sliced = "test_netcdf_rectilinear_column_sliced_$(Arch).nc"
    isfile(filepath_sliced) && rm(filepath_sliced)

    simulation.output_writers[:sliced] =
        NetCDFOutputWriter(model, fields(model),
            filename = filepath_sliced,
            indices = (:, :, k_slice),
            schedule = IterationInterval(1),
            array_type = Array{Float64},
            with_halos = false,
            include_grid_metrics = true,
            verbose = true
        )

    run!(simulation)

    # Test NetCDF output with halos
    ds_h = NCDataset(filepath_with_halos)

    dims = ("z_f", "z_c")
    not_dims = ("x_f", "x_c", "y_f", "y_c")

    metrics = ("dz_f", "dz_c")
    not_metrics = ("dx_f", "dx_c", "dy_f", "dy_c")

    vars = ("u", "v", "w", "T", "S")

    for var in (dims..., metrics..., vars...)
        @test haskey(ds_h, var)
        @test haskey(ds_h[var].attrib, "long_name")
        @test haskey(ds_h[var].attrib, "units")
        @test eltype(ds_h[var]) == Float64
    end

    for var in (not_dims..., not_metrics...)
        @test !haskey(ds_h, var)
    end

    @test dimsize(ds_h[:z_f]) == (z_f=N + 2H + 1,)
    @test dimsize(ds_h[:z_c]) == (z_c=N + 2H,)

    @test dimsize(ds_h[:dz_f]) == (z_f=N + 2H + 1,)
    @test dimsize(ds_h[:dz_c]) == (z_c=N + 2H,)

    @test dimsize(ds_h[:u]) == (z_c=N + 2H,     time=Nt + 1)
    @test dimsize(ds_h[:v]) == (z_c=N + 2H,     time=Nt + 1)
    @test dimsize(ds_h[:w]) == (z_f=N + 2H + 1, time=Nt + 1)
    @test dimsize(ds_h[:T]) == (z_c=N + 2H,     time=Nt + 1)
    @test dimsize(ds_h[:S]) == (z_c=N + 2H,     time=Nt + 1)

    close(ds_h)
    rm(filepath_with_halos)

    # Test NetCDF sliced output
    ds_s = NCDataset(filepath_sliced)

    for var in (dims..., metrics..., vars...)
        @test haskey(ds_s, var)
        @test haskey(ds_s[var].attrib, "long_name")
        @test haskey(ds_s[var].attrib, "units")
        @test eltype(ds_s[var]) == Float64
    end

    for var in (not_dims..., not_metrics...)
        @test !haskey(ds_s, var)
    end

    @test dimsize(ds_s[:z_f]) == (z_f=n,)
    @test dimsize(ds_s[:z_c]) == (z_c=n,)

    @test dimsize(ds_s[:dz_f]) == (z_f=n,)
    @test dimsize(ds_s[:dz_c]) == (z_c=n,)

    @test dimsize(ds_s[:u]) == (z_c=n, time=Nt + 1)
    @test dimsize(ds_s[:v]) == (z_c=n, time=Nt + 1)
    @test dimsize(ds_s[:w]) == (z_f=n, time=Nt + 1)
    @test dimsize(ds_s[:T]) == (z_c=n, time=Nt + 1)
    @test dimsize(ds_s[:S]) == (z_c=n, time=Nt + 1)

    close(ds_s)
    rm(filepath_sliced)

    return nothing
end

function test_thermal_bubble_netcdf_output(arch, FT)
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

    Arch = typeof(arch)
    nc_filepath = "test_thermal_bubble_$(Arch)_$FT.nc"
    isfile(nc_filepath) && rm(nc_filepath)

    nc_writer = NetCDFOutputWriter(model, outputs,
        filename = nc_filepath,
        schedule = IterationInterval(10),
        array_type = Array{FT},
        include_grid_metrics = false,
        verbose = true
    )

    push!(simulation.output_writers, nc_writer)

    i_slice = 1:10
    j_slice = 13
    k_slice = 9:11
    indices = (i_slice, j_slice, k_slice)
    j_slice = j_slice:j_slice  # So we can correctly index with it for later tests.

    nc_sliced_filepath = "test_thermal_bubble_sliced_$(Arch)_$FT.nc"
    isfile(nc_sliced_filepath) && rm(nc_sliced_filepath)

    nc_sliced_writer = NetCDFOutputWriter(model, outputs,
        filename = nc_sliced_filepath,
        schedule = IterationInterval(10),
        array_type = Array{FT},
        indices = indices,
        include_grid_metrics = false,
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

    @test eltype(ds3["time"]) == Float64

    @test eltype(ds3["x_c"]) == FT
    @test eltype(ds3["x_f"]) == FT
    @test eltype(ds3["y_c"]) == FT
    @test eltype(ds3["y_f"]) == FT
    @test eltype(ds3["z_c"]) == FT
    @test eltype(ds3["z_f"]) == FT

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

    @test eltype(ds3["u"]) == FT
    @test eltype(ds3["v"]) == FT
    @test eltype(ds3["w"]) == FT
    @test eltype(ds3["T"]) == FT
    @test eltype(ds3["S"]) == FT

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

    @test eltype(ds2["time"]) == Float64

    @test eltype(ds2["x_c"]) == FT
    @test eltype(ds2["x_f"]) == FT
    @test eltype(ds2["y_c"]) == FT
    @test eltype(ds2["y_f"]) == FT
    @test eltype(ds2["z_c"]) == FT
    @test eltype(ds2["z_f"]) == FT

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

    @test eltype(ds2["u"]) == FT
    @test eltype(ds2["v"]) == FT
    @test eltype(ds2["w"]) == FT
    @test eltype(ds2["T"]) == FT
    @test eltype(ds2["S"]) == FT

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

function test_thermal_bubble_netcdf_output_with_halos(arch, FT)
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

    Arch = typeof(arch)
    nc_filepath = "test_thermal_bubble_with_halos_$Arch.nc"

    nc_writer = NetCDFOutputWriter(model,
        merge(model.velocities, model.tracers),
        filename = nc_filepath,
        schedule = IterationInterval(10),
        array_type = Array{FT},
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

    @test eltype(ds["time"]) == Float64

    @test eltype(ds["x_c"]) == FT
    @test eltype(ds["x_f"]) == FT
    @test eltype(ds["y_c"]) == FT
    @test eltype(ds["y_f"]) == FT
    @test eltype(ds["z_c"]) == FT
    @test eltype(ds["z_f"]) == FT

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

    @test eltype(ds["u"]) == FT
    @test eltype(ds["v"]) == FT
    @test eltype(ds["w"]) == FT
    @test eltype(ds["T"]) == FT
    @test eltype(ds["S"]) == FT

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

    Arch = typeof(arch)
    ow = NetCDFOutputWriter(model, (; u=model.velocities.u);
        dir = ".",
        filename = "test_size_file_splitting_$Arch",
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
    @test filesize("test_size_file_splitting_$(Arch)_part1.nc") > 200KiB
    @test filesize("test_size_file_splitting_$(Arch)_part2.nc") > 200KiB
    @test filesize("test_size_file_splitting_$(Arch)_part3.nc") < 200KiB
    @test !isfile("test_size_file_splitting_$(Arch)_part4.nc")

    for n in string.(1:3)
        filename = "test_size_file_splitting_$(Arch)_part$n.nc"
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

    Arch = typeof(arch)
    ow = NetCDFOutputWriter(model, (; u=model.velocities.u);
        dir = ".",
        filename = "test_time_file_splitting_$Arch",
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
        filename = "test_time_file_splitting_$(Arch)_part$n.nc"
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
    rm("test_time_file_splitting_$(Arch)_part4.nc")

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

    Arch = typeof(arch)
    nc_filepath = "test_function_outputs_$Arch.nc"

    simulation.output_writers[:food] =
        NetCDFOutputWriter(model, outputs;
            global_attributes,
            output_attributes,
            filename = nc_filepath,
            schedule = TimeInterval(Δt),
            dimensions = dims,
            array_type = Array{Float64},
            include_grid_metrics = false,
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

    @test eltype(ds["time"]) == Float64

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

    Arch = typeof(arch)
    nc_filepath = "volume_averaged_field_test_$Arch.nc"

    simulation.output_writers[:averages] =
        NetCDFOutputWriter(model,
            (; ∫c_dx, ∫∫c_dxdy, ∫∫∫c_dxdydz),
            array_type = Array{Float64},
            verbose = true,
            filename = nc_filepath,
            schedule = IterationInterval(2),
            include_grid_metrics = false
        )

    run!(simulation)

    ds = NCDataset(nc_filepath)

    for (n, t) in enumerate(ds["time"])
        @test all(ds["∫c_dx"][:,:, n] .≈ 1)
        @test all(ds["∫∫c_dxdy"][:, n] .≈ 1)
        @test all(ds["∫∫∫c_dxdydz"][n] .≈ 1)
    end

    close(ds)
    rm(nc_filepath)

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
        # for (stride, rtol) in zip((1, 2), (1e-5, 1e-3))
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

            Arch = typeof(arch)
            horizontal_average_nc_filepath = "decay_averaged_field_test_$Arch.nc"

            simulation.output_writers[:horizontal_average] =
                NetCDFOutputWriter(
                    model,
                    nc_outputs,
                    array_type = Array{Float64},
                    verbose = true,
                    filename = horizontal_average_nc_filepath,
                    schedule = TimeInterval(10Δt),
                    include_grid_metrics = false
                )

            multiple_time_average_nc_filepath = "decay_windowed_time_average_test_$Arch.nc"
            single_time_average_nc_filepath = "single_decay_windowed_time_average_test_$Arch.nc"
            window = 6Δt

            single_nc_output = Dict("c1" => ∫c1_dxdy)

            simulation.output_writers[:single_output_time_average] =
                NetCDFOutputWriter(
                    model,
                    single_nc_output,
                    array_type = Array{Float64},
                    verbose = true,
                    filename = single_time_average_nc_filepath,
                    schedule = AveragedTimeInterval(10Δt; window, stride),
                    include_grid_metrics = false
                )

            simulation.output_writers[:multiple_output_time_average] =
                NetCDFOutputWriter(
                    model,
                    nc_outputs,
                    array_type = Array{Float64},
                    verbose = true,
                    filename = multiple_time_average_nc_filepath,
                    schedule = AveragedTimeInterval(10Δt; window, stride),
                    include_grid_metrics = false
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

    Arch = typeof(arch)
    test_filename1 = "test_output_alignment1_$Arch.nc"
    simulation.output_writers[:stuff] =
        NetCDFOutputWriter(model,
            model.velocities,
            filename = test_filename1,
            schedule = TimeInterval(7.3),
            include_grid_metrics = false
        )

    test_filename2 = "test_output_alignment2_$Arch.nc"
    simulation.output_writers[:something] =
        NetCDFOutputWriter(model,
            model.tracers,
            filename = test_filename2,
            schedule = TimeInterval(3.0),
            include_grid_metrics = false
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

function test_netcdf_output_just_particles(arch)
    grid = RectilinearGrid(arch;
        topology = (Periodic, Periodic, Bounded),
        size = (5, 5, 5),
        x = (-1, 1),
        y = (-1, 1),
        z = (-1, 0)
    )

    Np = 10
    xs = on_architecture(arch, 0.25 * ones(Np))
    ys = on_architecture(arch, -0.4 * ones(Np))
    zs = on_architecture(arch, -0.12 * ones(Np))

    particles = LagrangianParticles(x=xs, y=ys, z=zs)

    model = NonhydrostaticModel(; grid, particles)

    set!(model, u=1, v=1)

    Nt = 10
    simulation = Simulation(model, Δt=1e-2, stop_iteration=Nt)

    Arch = typeof(arch)
    filepath = "test_just_particles_$Arch.nc"
    simulation.output_writers[:particles_nc] =
        NetCDFOutputWriter(model,
            (; particles = model.particles),
            filename = filepath,
            schedule = IterationInterval(1),
            include_grid_metrics = false
        )

    run!(simulation)

    ds = NCDataset(filepath)

    @test haskey(ds, "time")
    @test length(ds[:time]) == Nt + 1

    @test haskey(ds, "particle_id")
    @test length(ds[:particle_id]) == Np

    @test haskey(ds, "x")
    @test haskey(ds, "y")
    @test haskey(ds, "z")

    @test size(ds[:x]) == (Np, Nt+1)
    @test size(ds[:y]) == (Np, Nt+1)
    @test size(ds[:z]) == (Np, Nt+1)

    close(ds)
    rm(filepath)

    return nothing
end

function test_netcdf_output_particles_and_fields(arch)
    N = 5

    grid = RectilinearGrid(arch;
        topology = (Periodic, Periodic, Bounded),
        size = (N, N, N),
        x = (-1, 1),
        y = (-1, 1),
        z = (-1, 0)
    )

    Np = 10
    xs = on_architecture(arch, 0.25 * ones(Np))
    ys = on_architecture(arch, -0.4 * ones(Np))
    zs = on_architecture(arch, -0.12 * ones(Np))

    particles = LagrangianParticles(x=xs, y=ys, z=zs)

    model = NonhydrostaticModel(; grid, particles)

    set!(model, u=1, v=1)

    Nt = 10
    simulation = Simulation(model, Δt=1e-2, stop_iteration=Nt)

    outputs = (
        particles = model.particles,
        u = model.velocities.u,
        v = model.velocities.v,
        w = model.velocities.w
    )

    Arch = typeof(arch)
    filepath = "test_particles_and_fields_$Arch.nc"
    simulation.output_writers[:particles_nc] =
        NetCDFOutputWriter(model, outputs,
            filename = filepath,
            schedule = IterationInterval(1),
            include_grid_metrics = false
        )

    run!(simulation)

    ds = NCDataset(filepath)

    @test haskey(ds, "time")
    @test length(ds[:time]) == Nt + 1

    @test haskey(ds, "particle_id")
    @test length(ds[:particle_id]) == Np

    @test haskey(ds, "x")
    @test haskey(ds, "y")
    @test haskey(ds, "z")

    @test size(ds[:x]) == (Np, Nt+1)
    @test size(ds[:y]) == (Np, Nt+1)
    @test size(ds[:z]) == (Np, Nt+1)

    @test haskey(ds, "u")
    @test haskey(ds, "v")
    @test haskey(ds, "w")

    @test size(ds[:u]) == (N, N, N,   Nt+1)
    @test size(ds[:v]) == (N, N, N,   Nt+1)
    @test size(ds[:w]) == (N, N, N+1, Nt+1)

    close(ds)
    rm(filepath)

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

    Arch = typeof(arch)
    nc_filepath = "test_netcdf_vertically_stretched_grid_output_$Arch.nc"

    simulation.output_writers[:fields] =
        NetCDFOutputWriter(model,
            merge(model.velocities, model.tracers),
            filename = nc_filepath,
            schedule = IterationInterval(1),
            array_type = Array{Float64},
            include_grid_metrics = false,
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

    Arch = typeof(arch)
    nc_filepath = "test_netcdf_regular_lat_lon_grid_output_$Arch.nc"

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

for arch in [CPU(), GPU()]
    @testset "NetCDF output writer [$(typeof(arch))]" begin
        @info "  Testing NetCDF output writer [$(typeof(arch))]..."

        test_datetime_netcdf_output(arch)
        test_timedate_netcdf_output(arch)

        test_netcdf_grid_metrics_rectilinear(arch, Float64)
        test_netcdf_grid_metrics_rectilinear(arch, Float32)

        test_netcdf_rectilinear_flat_xy(arch)
        test_netcdf_rectilinear_flat_xz(arch)
        test_netcdf_rectilinear_flat_yz(arch)

        test_thermal_bubble_netcdf_output(arch, Float64)
        test_thermal_bubble_netcdf_output(arch, Float32)
        test_thermal_bubble_netcdf_output_with_halos(arch, Float64)
        test_thermal_bubble_netcdf_output_with_halos(arch, Float32)

        test_netcdf_size_file_splitting(arch)
        test_netcdf_time_file_splitting(arch)

        test_netcdf_function_output(arch)
        test_netcdf_output_alignment(arch)

        test_netcdf_spatial_average(arch)
        test_netcdf_time_averaging(arch)

        test_netcdf_output_just_particles(arch)
        test_netcdf_output_particles_and_fields(arch)

        test_netcdf_vertically_stretched_grid_output(arch)

        # test_netcdf_regular_lat_lon_grid_output(arch; immersed = false)
        # test_netcdf_regular_lat_lon_grid_output(arch; immersed = true)
    end
end
