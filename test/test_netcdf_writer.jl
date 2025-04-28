include("dependencies_for_runtests.jl")

using TimesDates: TimeDate
using Dates: DateTime, Nanosecond, Millisecond
using TimesDates: TimeDate

using CUDA
using NCDatasets
using SeawaterPolynomials.TEOS10: TEOS10EquationOfState
using SeawaterPolynomials.SecondOrderSeawaterPolynomials: RoquetEquationOfState

using Oceananigans: Clock
using Oceananigans.Models.HydrostaticFreeSurfaceModels: VectorInvariant

function test_datetime_netcdf_output(arch)
    grid = RectilinearGrid(arch, size=(1, 1, 1), extent=(1, 1, 1))

    clock = Clock(time=DateTime(2021, 1, 1))

    model = NonhydrostaticModel(; grid, clock,
                                  timestepper = :QuasiAdamsBashforth2,
                                  buoyancy = SeawaterBuoyancy(),
                                  tracers=(:T, :S))

    Δt = 5days + 3hours + 44.123seconds
    simulation = Simulation(model; Δt, stop_time=DateTime(2021, 2, 1))

    Arch = typeof(arch)
    filepath = "test_datetime_$Arch.nc"
    isfile(filepath) && rm(filepath)

    simulation.output_writers[:netcdf] = NetCDFWriter(model, fields(model);
                                                      filename = filepath,
                                                      schedule = IterationInterval(1),
                                                      include_grid_metrics = false)

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
                                  tracers = (:T, :S))

    Δt = 5days + 3hours + 44.123seconds
    simulation = Simulation(model, Δt=Δt, stop_time=TimeDate(2021, 2, 1))

    Arch = typeof(arch)
    filepath = "test_timedate_$Arch.nc"
    isfile(filepath) && rm(filepath)

    simulation.output_writers[:netcdf] = NetCDFWriter(model, fields(model);
                                                      filename = filepath,
                                                      schedule = IterationInterval(1),
                                                      include_grid_metrics = false)

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
                           x = (0, 1), y = (0, 2), z = LinRange(0, 3, Nz + 1))

    model = NonhydrostaticModel(; grid,
                                  closure = ScalarDiffusivity(ν=4e-2, κ=4e-2),
                                  buoyancy = SeawaterBuoyancy(),
                                  tracers = (:T, :S))

    Nt = 10
    simulation = Simulation(model, Δt=0.1, stop_iteration=Nt)

    Arch = typeof(arch)
    filepath_metrics_halos = "test_grid_metrics_rectilinear_halos_$(Arch)_$FT.nc"
    isfile(filepath_metrics_halos) && rm(filepath_metrics_halos)

    simulation.output_writers[:with_metrics_and_halos] =
        NetCDFWriter(model, fields(model),
            filename = filepath_metrics_halos,
            schedule = IterationInterval(1),
            array_type = Array{FT},
            with_halos = true,
            include_grid_metrics = true,
            verbose = true)

    filepath_metrics_nohalos = "test_grid_metrics_rectilinear_nohalos_$(Arch)_$FT.nc"
    isfile(filepath_metrics_nohalos) && rm(filepath_metrics_nohalos)

    simulation.output_writers[:with_metrics_no_halos] = NetCDFWriter(model, fields(model),
                                                                     filename = filepath_metrics_nohalos,
                                                                     schedule = IterationInterval(1),
                                                                     array_type = Array{FT},
                                                                     with_halos = false,
                                                                     include_grid_metrics = true,
                                                                     verbose = true)

    filepath_nometrics = "test_grid_metrics_rectilinear_nometrics_$(Arch)_$FT.nc"
    isfile(filepath_nometrics) && rm(filepath_nometrics)

    simulation.output_writers[:no_metrics] = NetCDFWriter(model, fields(model),
                                                          filename = filepath_nometrics,
                                                          schedule = IterationInterval(1),
                                                          array_type = Array{FT},
                                                          with_halos = true,
                                                          include_grid_metrics = false,
                                                          verbose = true)

    i_slice = Colon()
    j_slice = 2:4
    k_slice = Nz

    nx = Nx
    ny = length(j_slice)
    nz = 1

    filepath_sliced = "test_grid_metrics_rectilinear_sliced_$(Arch)_$FT.nc"
    isfile(filepath_sliced) && rm(filepath_sliced)

    simulation.output_writers[:sliced] = NetCDFWriter(model, fields(model),
                                                      filename = filepath_sliced,
                                                      indices = (i_slice, j_slice, k_slice),
                                                      schedule = IterationInterval(1),
                                                      array_type = Array{FT},
                                                      with_halos = false,
                                                      include_grid_metrics = true,
                                                      verbose = true)

    run!(simulation)

    # Test NetCDF output with metrics and halos
    ds_mh = NCDataset(filepath_metrics_halos)

    @test haskey(ds_mh, "time")
    @test eltype(ds_mh["time"]) == Float64

    dims = ("x_faa", "x_caa", "y_afa", "y_aca", "z_aaf", "z_aac")
    metrics = ("Δx_faa", "Δx_caa", "Δy_afa", "Δy_aca", "Δy_afa", "Δy_aca")
    vars = ("u", "v", "w", "T", "S")

    for var in (dims..., metrics..., vars...)
        @test haskey(ds_mh, var)
        @test haskey(ds_mh[var].attrib, "long_name")
        @test haskey(ds_mh[var].attrib, "units")
        @test eltype(ds_mh[var]) == FT
    end

    @test dimsize(ds_mh["time"]) == (time=Nt + 1,)

    @test dimsize(ds_mh[:x_faa]) == (x_faa=Nx + 2Hx,)
    @test dimsize(ds_mh[:x_caa]) == (x_caa=Nx + 2Hx,)
    @test dimsize(ds_mh[:y_afa]) == (y_afa=Ny + 2Hy + 1,)
    @test dimsize(ds_mh[:y_aca]) == (y_aca=Ny + 2Hy,)
    @test dimsize(ds_mh[:z_aaf]) == (z_aaf=Nz + 2Hz + 1,)
    @test dimsize(ds_mh[:z_aac]) == (z_aac=Nz + 2Hz,)

    @test dimsize(ds_mh[:Δx_faa]) == (x_faa=Nx + 2Hx,)
    @test dimsize(ds_mh[:Δx_caa]) == (x_caa=Nx + 2Hx,)
    @test dimsize(ds_mh[:Δy_afa]) == (y_afa=Ny + 2Hy + 1,)
    @test dimsize(ds_mh[:Δy_aca]) == (y_aca=Ny + 2Hy,)
    @test dimsize(ds_mh[:Δz_aaf]) == (z_aaf=Nz + 2Hz + 1,)
    @test dimsize(ds_mh[:Δz_aac]) == (z_aac=Nz + 2Hz,)

    @test dimsize(ds_mh[:u]) == (x_faa=Nx + 2Hx, y_aca=Ny + 2Hy,     z_aac=Nz + 2Hz,     time=Nt + 1)
    @test dimsize(ds_mh[:v]) == (x_caa=Nx + 2Hx, y_afa=Ny + 2Hy + 1, z_aac=Nz + 2Hz,     time=Nt + 1)
    @test dimsize(ds_mh[:w]) == (x_caa=Nx + 2Hx, y_aca=Ny + 2Hy,     z_aaf=Nz + 2Hz + 1, time=Nt + 1)
    @test dimsize(ds_mh[:T]) == (x_caa=Nx + 2Hx, y_aca=Ny + 2Hy,     z_aac=Nz + 2Hz,     time=Nt + 1)
    @test dimsize(ds_mh[:S]) == (x_caa=Nx + 2Hx, y_aca=Ny + 2Hy,     z_aac=Nz + 2Hz,     time=Nt + 1)

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

    @test dimsize(ds_m[:x_faa]) == (x_faa=Nx,)
    @test dimsize(ds_m[:x_caa]) == (x_caa=Nx,)
    @test dimsize(ds_m[:y_afa]) == (y_afa=Ny + 1,)
    @test dimsize(ds_m[:y_aca]) == (y_aca=Ny,)
    @test dimsize(ds_m[:z_aaf]) == (z_aaf=Nz + 1,)
    @test dimsize(ds_m[:z_aac]) == (z_aac=Nz,)

    @test dimsize(ds_m[:Δx_faa]) == (x_faa=Nx,)
    @test dimsize(ds_m[:Δx_caa]) == (x_caa=Nx,)
    @test dimsize(ds_m[:Δy_afa]) == (y_afa=Ny + 1,)
    @test dimsize(ds_m[:Δy_aca]) == (y_aca=Ny,)
    @test dimsize(ds_m[:Δz_aaf]) == (z_aaf=Nz + 1,)
    @test dimsize(ds_m[:Δz_aac]) == (z_aac=Nz,)

    @test dimsize(ds_m[:u]) == (x_faa=Nx, y_aca=Ny,     z_aac=Nz,     time=Nt + 1)
    @test dimsize(ds_m[:v]) == (x_caa=Nx, y_afa=Ny + 1, z_aac=Nz,     time=Nt + 1)
    @test dimsize(ds_m[:w]) == (x_caa=Nx, y_aca=Ny,     z_aaf=Nz + 1, time=Nt + 1)
    @test dimsize(ds_m[:T]) == (x_caa=Nx, y_aca=Ny,     z_aac=Nz,     time=Nt + 1)
    @test dimsize(ds_m[:S]) == (x_caa=Nx, y_aca=Ny,     z_aac=Nz,     time=Nt + 1)

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

    @test dimsize(ds_h[:x_faa]) == (x_faa=Nx + 2Hx,)
    @test dimsize(ds_h[:x_caa]) == (x_caa=Nx + 2Hx,)
    @test dimsize(ds_h[:y_afa]) == (y_afa=Ny + 2Hy + 1,)
    @test dimsize(ds_h[:y_aca]) == (y_aca=Ny + 2Hy,)
    @test dimsize(ds_h[:z_aaf]) == (z_aaf=Nz + 2Hz + 1,)
    @test dimsize(ds_h[:z_aac]) == (z_aac=Nz + 2Hz,)

    @test dimsize(ds_h[:u]) == (x_faa=Nx + 2Hx, y_aca=Ny + 2Hy,     z_aac=Nz + 2Hz,     time=Nt + 1)
    @test dimsize(ds_h[:v]) == (x_caa=Nx + 2Hx, y_afa=Ny + 2Hy + 1, z_aac=Nz + 2Hz,     time=Nt + 1)
    @test dimsize(ds_h[:w]) == (x_caa=Nx + 2Hx, y_aca=Ny + 2Hy,     z_aaf=Nz + 2Hz + 1, time=Nt + 1)
    @test dimsize(ds_h[:T]) == (x_caa=Nx + 2Hx, y_aca=Ny + 2Hy,     z_aac=Nz + 2Hz,     time=Nt + 1)
    @test dimsize(ds_h[:S]) == (x_caa=Nx + 2Hx, y_aca=Ny + 2Hy,     z_aac=Nz + 2Hz,     time=Nt + 1)

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

    @test dimsize(ds_s[:x_faa]) == (x_faa=nx,)
    @test dimsize(ds_s[:x_caa]) == (x_caa=nx,)
    @test dimsize(ds_s[:y_afa]) == (y_afa=ny,)
    @test dimsize(ds_s[:y_aca]) == (y_aca=ny,)
    @test dimsize(ds_s[:z_aaf]) == (z_aaf=nz,)
    @test dimsize(ds_s[:z_aac]) == (z_aac=nz,)

    @test dimsize(ds_s[:Δx_faa]) == (x_faa=nx,)
    @test dimsize(ds_s[:Δx_caa]) == (x_caa=nx,)
    @test dimsize(ds_s[:Δy_afa]) == (y_afa=ny,)
    @test dimsize(ds_s[:Δy_aca]) == (y_aca=ny,)
    @test dimsize(ds_s[:Δz_aaf]) == (z_aaf=nz,)
    @test dimsize(ds_s[:Δz_aac]) == (z_aac=nz,)

    @test dimsize(ds_s[:u]) == (x_faa=nx, y_aca=ny, z_aac=nz, time=Nt + 1)
    @test dimsize(ds_s[:v]) == (x_caa=nx, y_afa=ny, z_aac=nz, time=Nt + 1)
    @test dimsize(ds_s[:w]) == (x_caa=nx, y_aca=ny, z_aaf=nz, time=Nt + 1)
    @test dimsize(ds_s[:T]) == (x_caa=nx, y_aca=ny, z_aac=nz, time=Nt + 1)
    @test dimsize(ds_s[:S]) == (x_caa=nx, y_aca=ny, z_aac=nz, time=Nt + 1)

    close(ds_s)
    rm(filepath_sliced)

    return nothing
end

function test_netcdf_grid_metrics_latlon(arch, FT)
    Nλ, Nφ, Nz = 10, 9, 8
    Hλ, Hφ, Hz = 4, 3, 2

    grid = LatitudeLongitudeGrid(arch,
                                 topology = (Bounded, Bounded, Bounded),
                                 size = (Nλ, Nφ, Nz),
                                 halo = (Hλ, Hφ, Hz),
                                 longitude = (-15, 15),
                                 latitude = (-10, 10),
                                 z = LinRange(-1000, 0, Nz + 1))

    model = HydrostaticFreeSurfaceModel(; grid,
                                          momentum_advection = VectorInvariant(),
                                          buoyancy = SeawaterBuoyancy(),
                                          tracers = (:T, :S))

    Nt = 5
    simulation = Simulation(model, Δt=0.1, stop_iteration=Nt)

    outputs = merge(model.velocities, model.tracers)

    Arch = typeof(arch)
    filepath_metrics_halos = "test_grid_metrics_latlon_halos_$(Arch)_$FT.nc"
    isfile(filepath_metrics_halos) && rm(filepath_metrics_halos)

    # Test with halos and metrics
    simulation.output_writers[:with_metrics_and_halos] = NetCDFWriter(model, outputs,
                                                                      filename = filepath_metrics_halos,
                                                                      schedule = IterationInterval(1),
                                                                      array_type = Array{FT},
                                                                      with_halos = true,
                                                                      include_grid_metrics = true,
                                                                      verbose = true)

    # Test with halos but no metrics
    filepath_nometrics = "test_grid_metrics_latlon_nometrics_$(Arch)_$FT.nc"
    isfile(filepath_nometrics) && rm(filepath_nometrics)

    simulation.output_writers[:no_metrics] = NetCDFWriter(model, outputs,
                                                          filename = filepath_nometrics,
                                                          schedule = IterationInterval(1),
                                                          array_type = Array{FT},
                                                          with_halos = true,
                                                          include_grid_metrics = false,
                                                          verbose = true)

    # Test without halos but with metrics
    filepath_metrics_nohalos = "test_grid_metrics_latlon_nohalos_$(Arch)_$FT.nc"
    isfile(filepath_metrics_nohalos) && rm(filepath_metrics_nohalos)

    simulation.output_writers[:with_metrics_no_halos] = NetCDFWriter(model, outputs,
                                                                     filename = filepath_metrics_nohalos,
                                                                     schedule = IterationInterval(1),
                                                                     array_type = Array{FT},
                                                                     with_halos = false,
                                                                     include_grid_metrics = true,
                                                                     verbose = true)

    # Test a slice of the domain
    i_slice = Colon()
    j_slice = 3:7
    k_slice = Nz

    nx = Nλ
    ny = length(j_slice)
    nz = 1

    filepath_sliced = "test_grid_metrics_latlon_sliced_$(Arch)_$FT.nc"
    isfile(filepath_sliced) && rm(filepath_sliced)

    simulation.output_writers[:sliced] = NetCDFWriter(model, outputs,
                                                      filename = filepath_sliced,
                                                      indices = (i_slice, j_slice, k_slice),
                                                      schedule = IterationInterval(1),
                                                      array_type = Array{FT},
                                                      with_halos = false,
                                                      include_grid_metrics = true,
                                                      verbose = true)

    run!(simulation)

    # Test NetCDF output with metrics and halos
    ds_mh = NCDataset(filepath_metrics_halos)

    @test haskey(ds_mh, "time")
    @test eltype(ds_mh["time"]) == Float64

    dims = ("λ_faa", "λ_caa", "φ_afa", "φ_aca", "z_aaf", "z_aac")
    metrics = ("Δλ_faa", "Δλ_caa", "Δλ_afa", "Δλ_aca", "Δz_aaf", "Δz_aac",
               "Δx_ffa", "Δx_fca", "Δx_cfa", "Δx_cca",
               "Δy_ffa", "Δy_fca", "Δy_cfa", "Δy_cca")
    vars = ("u", "v", "w", "T", "S")

    for var in (dims..., metrics..., vars...)
        @test haskey(ds_mh, var)
        @test haskey(ds_mh[var].attrib, "long_name")
        @test haskey(ds_mh[var].attrib, "units")
        @test eltype(ds_mh[var]) == FT
    end

    @test dimsize(ds_mh["time"]) == (time=Nt + 1,)

    @test dimsize(ds_mh[:λ_faa]) == (λ_faa=Nλ + 2Hλ + 1,)
    @test dimsize(ds_mh[:λ_caa]) == (λ_caa=Nλ + 2Hλ,)
    @test dimsize(ds_mh[:φ_afa]) == (φ_afa=Nφ + 2Hφ + 1,)
    @test dimsize(ds_mh[:φ_aca]) == (φ_aca=Nφ + 2Hφ,)
    @test dimsize(ds_mh[:z_aaf]) == (z_aaf=Nz + 2Hz + 1,)
    @test dimsize(ds_mh[:z_aac]) == (z_aac=Nz + 2Hz,)

    @test dimsize(ds_mh[:Δλ_faa]) == (λ_faa=Nλ + 2Hλ + 1,)
    @test dimsize(ds_mh[:Δλ_caa]) == (λ_caa=Nλ + 2Hλ,)
    @test dimsize(ds_mh[:Δλ_afa]) == (φ_afa=Nφ + 2Hφ + 1,)
    @test dimsize(ds_mh[:Δλ_aca]) == (φ_aca=Nφ + 2Hφ,)
    @test dimsize(ds_mh[:Δz_aaf]) == (z_aaf=Nz + 2Hz + 1,)
    @test dimsize(ds_mh[:Δz_aac]) == (z_aac=Nz + 2Hz,)

    @test dimsize(ds_mh[:Δx_ffa]) == (λ_faa=Nλ + 2Hλ + 1, φ_afa=Nφ + 2Hφ + 1)
    @test dimsize(ds_mh[:Δx_fca]) == (λ_faa=Nλ + 2Hλ + 1, φ_aca=Nφ + 2Hφ)
    @test dimsize(ds_mh[:Δx_cfa]) == (λ_caa=Nλ + 2Hλ,     φ_afa=Nφ + 2Hφ + 1)
    @test dimsize(ds_mh[:Δx_cca]) == (λ_caa=Nλ + 2Hλ,     φ_aca=Nφ + 2Hφ)

    @test dimsize(ds_mh[:Δy_ffa]) == (λ_faa=Nλ + 2Hλ + 1, φ_afa=Nφ + 2Hφ + 1)
    @test dimsize(ds_mh[:Δy_fca]) == (λ_faa=Nλ + 2Hλ + 1, φ_aca=Nφ + 2Hφ)
    @test dimsize(ds_mh[:Δy_cfa]) == (λ_caa=Nλ + 2Hλ,     φ_afa=Nφ + 2Hφ + 1)
    @test dimsize(ds_mh[:Δy_cca]) == (λ_caa=Nλ + 2Hλ,     φ_aca=Nφ + 2Hφ)

    @test dimsize(ds_mh[:u]) == (λ_faa=Nλ + 2Hλ + 1, φ_aca=Nφ + 2Hφ,     z_aac=Nz + 2Hz,     time=Nt + 1)
    @test dimsize(ds_mh[:v]) == (λ_caa=Nλ + 2Hλ,     φ_afa=Nφ + 2Hφ + 1, z_aac=Nz + 2Hz,     time=Nt + 1)
    @test dimsize(ds_mh[:w]) == (λ_caa=Nλ + 2Hλ,     φ_aca=Nφ + 2Hφ,     z_aaf=Nz + 2Hz + 1, time=Nt + 1)
    @test dimsize(ds_mh[:T]) == (λ_caa=Nλ + 2Hλ,     φ_aca=Nφ + 2Hφ,     z_aac=Nz + 2Hz,     time=Nt + 1)
    @test dimsize(ds_mh[:S]) == (λ_caa=Nλ + 2Hλ,     φ_aca=Nφ + 2Hφ,     z_aac=Nz + 2Hz,     time=Nt + 1)

    close(ds_mh)
    rm(filepath_metrics_halos)

    # Test NetCDF output with halos but no metrics
    ds_h = NCDataset(filepath_nometrics)

    @test haskey(ds_h, "time")
    @test eltype(ds_h["time"]) == Float64

    for var in (dims..., vars...)
        @test haskey(ds_h, var)
        @test haskey(ds_h[var].attrib, "long_name")
        @test haskey(ds_h[var].attrib, "units")
        @test eltype(ds_h[var]) == FT
    end

    # Verify that metrics are not present
    for metric in metrics
        @test !haskey(ds_h, metric)
    end

    @test dimsize(ds_h["time"]) == (time=Nt + 1,)

    @test dimsize(ds_h[:λ_faa]) == (λ_faa=Nλ + 2Hλ + 1,)
    @test dimsize(ds_h[:λ_caa]) == (λ_caa=Nλ + 2Hλ,)
    @test dimsize(ds_h[:φ_afa]) == (φ_afa=Nφ + 2Hφ + 1,)
    @test dimsize(ds_h[:φ_aca]) == (φ_aca=Nφ + 2Hφ,)
    @test dimsize(ds_h[:z_aaf]) == (z_aaf=Nz + 2Hz + 1,)
    @test dimsize(ds_h[:z_aac]) == (z_aac=Nz + 2Hz,)

    @test dimsize(ds_h[:u]) == (λ_faa=Nλ + 2Hλ + 1, φ_aca=Nφ + 2Hφ,     z_aac=Nz + 2Hz,     time=Nt + 1)
    @test dimsize(ds_h[:v]) == (λ_caa=Nλ + 2Hλ,     φ_afa=Nφ + 2Hφ + 1, z_aac=Nz + 2Hz,     time=Nt + 1)
    @test dimsize(ds_h[:w]) == (λ_caa=Nλ + 2Hλ,     φ_aca=Nφ + 2Hφ,     z_aaf=Nz + 2Hz + 1, time=Nt + 1)
    @test dimsize(ds_h[:T]) == (λ_caa=Nλ + 2Hλ,     φ_aca=Nφ + 2Hφ,     z_aac=Nz + 2Hz,     time=Nt + 1)
    @test dimsize(ds_h[:S]) == (λ_caa=Nλ + 2Hλ,     φ_aca=Nφ + 2Hφ,     z_aac=Nz + 2Hz,     time=Nt + 1)

    close(ds_h)
    rm(filepath_nometrics)

    # Test NetCDF output with metrics but no halos
    ds_m = NCDataset(filepath_metrics_nohalos)

    for var in (dims..., metrics..., vars...)
        @test haskey(ds_m, var)
        @test haskey(ds_m[var].attrib, "long_name")
        @test haskey(ds_m[var].attrib, "units")
        @test eltype(ds_m[var]) == FT
    end

    @test dimsize(ds_m[:λ_faa]) == (λ_faa=Nλ + 1,)
    @test dimsize(ds_m[:λ_caa]) == (λ_caa=Nλ,)
    @test dimsize(ds_m[:φ_afa]) == (φ_afa=Nφ + 1,)
    @test dimsize(ds_m[:φ_aca]) == (φ_aca=Nφ,)
    @test dimsize(ds_m[:z_aaf]) == (z_aaf=Nz + 1,)
    @test dimsize(ds_m[:z_aac]) == (z_aac=Nz,)

    @test dimsize(ds_m[:Δλ_faa]) == (λ_faa=Nλ + 1,)
    @test dimsize(ds_m[:Δλ_caa]) == (λ_caa=Nλ,)
    @test dimsize(ds_m[:Δλ_afa]) == (φ_afa=Nφ + 1,)
    @test dimsize(ds_m[:Δλ_aca]) == (φ_aca=Nφ,)
    @test dimsize(ds_m[:Δz_aaf]) == (z_aaf=Nz + 1,)
    @test dimsize(ds_m[:Δz_aac]) == (z_aac=Nz,)

    @test dimsize(ds_m[:Δx_ffa]) == (λ_faa=Nλ + 1, φ_afa=Nφ + 1)
    @test dimsize(ds_m[:Δx_fca]) == (λ_faa=Nλ + 1, φ_aca=Nφ)
    @test dimsize(ds_m[:Δx_cfa]) == (λ_caa=Nλ,     φ_afa=Nφ + 1)
    @test dimsize(ds_m[:Δx_cca]) == (λ_caa=Nλ,     φ_aca=Nφ)

    @test dimsize(ds_m[:Δy_ffa]) == (λ_faa=Nλ + 1, φ_afa=Nφ + 1)
    @test dimsize(ds_m[:Δy_fca]) == (λ_faa=Nλ + 1, φ_aca=Nφ)
    @test dimsize(ds_m[:Δy_cfa]) == (λ_caa=Nλ,     φ_afa=Nφ + 1)
    @test dimsize(ds_m[:Δy_cca]) == (λ_caa=Nλ,     φ_aca=Nφ)

    @test dimsize(ds_m[:u]) == (λ_faa=Nλ + 1, φ_aca=Nφ,     z_aac=Nz,     time=Nt + 1)
    @test dimsize(ds_m[:v]) == (λ_caa=Nλ,     φ_afa=Nφ + 1, z_aac=Nz,     time=Nt + 1)
    @test dimsize(ds_m[:w]) == (λ_caa=Nλ,     φ_aca=Nφ,     z_aaf=Nz + 1, time=Nt + 1)
    @test dimsize(ds_m[:T]) == (λ_caa=Nλ,     φ_aca=Nφ,     z_aac=Nz,     time=Nt + 1)
    @test dimsize(ds_m[:S]) == (λ_caa=Nλ,     φ_aca=Nφ,     z_aac=Nz,     time=Nt + 1)

    close(ds_m)
    rm(filepath_metrics_nohalos)

    # Test NetCDF sliced output with metrics
    ds_s = NCDataset(filepath_sliced)

    for var in (dims..., metrics..., vars...)
        @test haskey(ds_s, var)
        @test haskey(ds_s[var].attrib, "long_name")
        @test haskey(ds_s[var].attrib, "units")
        @test eltype(ds_s[var]) == FT
    end

    @test dimsize(ds_s[:λ_faa]) == (λ_faa=nx + 1,)
    @test dimsize(ds_s[:λ_caa]) == (λ_caa=nx,)
    @test dimsize(ds_s[:φ_afa]) == (φ_afa=ny,)
    @test dimsize(ds_s[:φ_aca]) == (φ_aca=ny,)
    @test dimsize(ds_s[:z_aaf]) == (z_aaf=nz,)
    @test dimsize(ds_s[:z_aac]) == (z_aac=nz,)

    @test dimsize(ds_s[:Δλ_faa]) == (λ_faa=nx + 1,)
    @test dimsize(ds_s[:Δλ_caa]) == (λ_caa=nx,)
    @test dimsize(ds_s[:Δλ_afa]) == (φ_afa=ny,)
    @test dimsize(ds_s[:Δλ_aca]) == (φ_aca=ny,)
    @test dimsize(ds_s[:Δz_aaf]) == (z_aaf=nz,)
    @test dimsize(ds_s[:Δz_aac]) == (z_aac=nz,)

    @test dimsize(ds_s[:Δx_ffa]) == (λ_faa=nx + 1, φ_afa=ny)
    @test dimsize(ds_s[:Δx_fca]) == (λ_faa=nx + 1, φ_aca=ny)
    @test dimsize(ds_s[:Δx_cfa]) == (λ_caa=nx,     φ_afa=ny)
    @test dimsize(ds_s[:Δx_cca]) == (λ_caa=nx,     φ_aca=ny)

    @test dimsize(ds_s[:Δy_ffa]) == (λ_faa=nx + 1, φ_afa=ny)
    @test dimsize(ds_s[:Δy_fca]) == (λ_faa=nx + 1, φ_aca=ny)
    @test dimsize(ds_s[:Δy_cfa]) == (λ_caa=nx,     φ_afa=ny)
    @test dimsize(ds_s[:Δy_cca]) == (λ_caa=nx,     φ_aca=ny)

    @test dimsize(ds_s[:u]) == (λ_faa=nx + 1, φ_aca=ny, z_aac=nz, time=Nt + 1)
    @test dimsize(ds_s[:v]) == (λ_caa=nx,     φ_afa=ny, z_aac=nz, time=Nt + 1)
    @test dimsize(ds_s[:w]) == (λ_caa=nx,     φ_aca=ny, z_aaf=nz, time=Nt + 1)
    @test dimsize(ds_s[:T]) == (λ_caa=nx,     φ_aca=ny, z_aac=nz, time=Nt + 1)
    @test dimsize(ds_s[:S]) == (λ_caa=nx,     φ_aca=ny, z_aac=nz, time=Nt + 1)

    close(ds_s)
    rm(filepath_sliced)

    return nothing
end

function test_netcdf_rectilinear_grid_fitted_bottom(arch)
    Nx, Ny, Nz = 16, 16, 16
    Hx, Hy, Hz = 2, 3, 4

    Lx, Ly, H = 1, 1, 1

    underlying_grid = RectilinearGrid(arch;
                                      topology = (Bounded, Bounded, Bounded),
                                      size = (Nx, Ny, Nz),
                                      halo = (Hx, Hy, Hz),
                                      x = (-Lx, Lx),
                                      y = (-Ly, Ly),
                                      z = (-H, 0))

    height = H / 2
    width = Lx / 3
    mount(x, y) = height * exp(-x^2 / 2width^2) * exp(-y^2 / 2width^2)
    bottom(x, y) = -H + mount(x, y)

    grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom))

    model = NonhydrostaticModel(; grid,
                                  closure = ScalarDiffusivity(ν=4e-2, κ=4e-2),
                                  buoyancy = SeawaterBuoyancy(),
                                  tracers = (:T, :S))

    Nt = 10
    simulation = Simulation(model, Δt=0.1, stop_iteration=Nt)

    Arch = typeof(arch)
    filepath_with_halos = "test_immersed_grid_rectilinear_with_halos_$Arch.nc"
    isfile(filepath_with_halos) && rm(filepath_with_halos)

    simulation.output_writers[:with_halos] = NetCDFWriter(model, fields(model),
                                                          filename = filepath_with_halos,
                                                          schedule = IterationInterval(1),
                                                          array_type = Array{Float64},
                                                          with_halos = true,
                                                          include_grid_metrics = true,
                                                          verbose = true)

    filepath_no_halos = "test_immersed_grid_rectilinear_no_halos_$Arch.nc"
    isfile(filepath_no_halos) && rm(filepath_no_halos)

    simulation.output_writers[:no_halos] = NetCDFWriter(model, fields(model),
                                                        filename = filepath_no_halos,
                                                        schedule = IterationInterval(1),
                                                        array_type = Array{Float32},
                                                        with_halos = false,
                                                        include_grid_metrics = true,
                                                        verbose = true)

    filepath_sliced = "test_immersed_grid_rectilinear_sliced_$Arch.nc"
    isfile(filepath_sliced) && rm(filepath_sliced)

    i_slice = 4:12
    j_slice = 6:8
    k_slice = Nz

    nx = length(i_slice)
    ny = length(j_slice)
    nz = 1

    simulation.output_writers[:sliced] = NetCDFWriter(model, fields(model),
                                                      filename = filepath_sliced,
                                                      schedule = IterationInterval(1),
                                                      array_type = Array{Float32},
                                                      indices = (i_slice, j_slice, k_slice),
                                                      with_halos = false,
                                                      include_grid_metrics = true,
                                                      verbose = true)

    run!(simulation)

    # Test NetCDF output with halos
    ds_h = NCDataset(filepath_with_halos)

    @test haskey(ds_h, "bottom_height")
    @test eltype(ds_h[:bottom_height]) == Float64
    @test dimsize(ds_h[:bottom_height]) == (x_caa=Nx + 2Hx, y_aca=Ny + 2Hy)

    for loc in ("ccc", "fcc", "cfc", "ccf")
        @test haskey(ds_h, "immersed_boundary_mask_$loc")
        @test eltype(ds_h["immersed_boundary_mask_$loc"]) == Float64
    end

    @test dimsize(ds_h[:immersed_boundary_mask_ccc]) == (x_caa=Nx + 2Hx,     y_aca=Ny + 2Hy,     z_aac=Nz + 2Hz)
    @test dimsize(ds_h[:immersed_boundary_mask_fcc]) == (x_faa=Nx + 2Hx + 1, y_aca=Ny + 2Hy,     z_aac=Nz + 2Hz)
    @test dimsize(ds_h[:immersed_boundary_mask_cfc]) == (x_caa=Nx + 2Hx,     y_afa=Ny + 2Hy + 1, z_aac=Nz + 2Hz)
    @test dimsize(ds_h[:immersed_boundary_mask_ccf]) == (x_caa=Nx + 2Hx,     y_aca=Ny + 2Hy,     z_aaf=Nz + 2Hz + 1)

    @test all(ds_h[:bottom_height][:, :] .≈ Array(parent(grid.immersed_boundary.bottom_height)))

    close(ds_h)
    rm(filepath_with_halos)

    # Test NetCDF output without halos
    ds_n = NCDataset(filepath_no_halos)

    @test haskey(ds_n, "bottom_height")
    @test eltype(ds_n[:bottom_height]) == Float32
    @test dimsize(ds_n[:bottom_height]) == (x_caa=Nx, y_aca=Ny)

    for loc in ("ccc", "fcc", "cfc", "ccf")
        @test haskey(ds_n, "immersed_boundary_mask_$loc")
        @test eltype(ds_n["immersed_boundary_mask_$loc"]) == Float32
    end

    @test dimsize(ds_n[:immersed_boundary_mask_ccc]) == (x_caa=Nx,     y_aca=Ny,     z_aac=Nz)
    @test dimsize(ds_n[:immersed_boundary_mask_fcc]) == (x_faa=Nx + 1, y_aca=Ny,     z_aac=Nz)
    @test dimsize(ds_n[:immersed_boundary_mask_cfc]) == (x_caa=Nx,     y_afa=Ny + 1, z_aac=Nz)
    @test dimsize(ds_n[:immersed_boundary_mask_ccf]) == (x_caa=Nx,     y_aca=Ny,     z_aaf=Nz + 1)

    @test all(ds_n[:bottom_height][:, :] .≈ Array(interior(grid.immersed_boundary.bottom_height)))

    close(ds_n)
    rm(filepath_no_halos)

    # Test NetCDF sliced output
    ds_s = NCDataset(filepath_sliced)

    @test haskey(ds_s, "bottom_height")
    @test eltype(ds_s[:bottom_height]) == Float32
    @test dimsize(ds_s[:bottom_height]) == (x_caa=nx, y_aca=ny)

    for loc in ("ccc", "fcc", "cfc", "ccf")
        @test haskey(ds_s, "immersed_boundary_mask_$loc")
        @test eltype(ds_s["immersed_boundary_mask_$loc"]) == Float32
    end

    @test dimsize(ds_s[:immersed_boundary_mask_ccc]) == (x_caa=nx, y_aca=ny, z_aac=nz)
    @test dimsize(ds_s[:immersed_boundary_mask_fcc]) == (x_faa=nx, y_aca=ny, z_aac=nz)
    @test dimsize(ds_s[:immersed_boundary_mask_cfc]) == (x_caa=nx, y_afa=ny, z_aac=nz)
    @test dimsize(ds_s[:immersed_boundary_mask_ccf]) == (x_caa=nx, y_aca=ny, z_aaf=nz)

    @test all(ds_s[:bottom_height][:, :] .≈ Array(interior(grid.immersed_boundary.bottom_height, i_slice, j_slice)))

    close(ds_s)
    rm(filepath_sliced)

    return nothing
end

function test_netcdf_latlon_grid_fitted_bottom(arch)
    Nλ, Nφ, Nz = 16, 16, 16
    Hλ, Hφ, Hz = 2, 3, 4
    Lλ, Lφ, H = 20, 10, 1000

    underlying_grid = LatitudeLongitudeGrid(arch;
                                            topology = (Bounded, Bounded, Bounded),
                                            size = (Nλ, Nφ, Nz),
                                            halo = (Hλ, Hφ, Hz),
                                            longitude = (-Lλ, Lλ),
                                            latitude = (-Lφ, Lφ),
                                            z = (-H, 0))

    # Create a Gaussian seamount
    height = H / 2
    λ_width = Lλ / 3
    φ_width = Lφ / 3
    seamount(λ, φ) = height * exp(-λ^2 / 2λ_width^2) * exp(-φ^2 / 2φ_width^2)
    bottom(λ, φ) = -H + seamount(λ, φ)

    grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom))

    model = HydrostaticFreeSurfaceModel(; grid,
                                          momentum_advection = VectorInvariant(),
                                          buoyancy = SeawaterBuoyancy(),
                                          tracers = (:T, :S))

    Nt = 5
    simulation = Simulation(model, Δt=0.1, stop_iteration=Nt)

    outputs = merge(model.velocities, model.tracers)

    Arch = typeof(arch)
    filepath_with_halos = "test_immersed_grid_latlon_with_halos_$Arch.nc"
    isfile(filepath_with_halos) && rm(filepath_with_halos)

    # Test with halos with Float64 output
    simulation.output_writers[:with_halos] = NetCDFWriter(model, outputs,
                                                          filename = filepath_with_halos,
                                                          schedule = IterationInterval(1),
                                                          array_type = Array{Float64},
                                                          with_halos = true,
                                                          include_grid_metrics = true,
                                                          verbose = true)

    # Test without halos with Float32 output
    filepath_no_halos = "test_immersed_grid_latlon_no_halos_$Arch.nc"
    isfile(filepath_no_halos) && rm(filepath_no_halos)

    simulation.output_writers[:no_halos] = NetCDFWriter(model, outputs,
                                                        filename = filepath_no_halos,
                                                        schedule = IterationInterval(1),
                                                        array_type = Array{Float32},
                                                        with_halos = false,
                                                        include_grid_metrics = true,
                                                        verbose = true)

    # Test with slice
    filepath_sliced = "test_immersed_grid_latlon_sliced_$Arch.nc"
    isfile(filepath_sliced) && rm(filepath_sliced)

    i_slice = 4:12
    j_slice = 6:8
    k_slice = Nz

    nλ = length(i_slice)
    nφ = length(j_slice)
    nz = 1

    simulation.output_writers[:sliced] = NetCDFWriter(model, outputs,
                                                      filename = filepath_sliced,
                                                      schedule = IterationInterval(1),
                                                      array_type = Array{Float32},
                                                      indices = (i_slice, j_slice, k_slice),
                                                      with_halos = false,
                                                      include_grid_metrics = true,
                                                      verbose = true)

    run!(simulation)

    # Test NetCDF output with halos
    ds_h = NCDataset(filepath_with_halos)

    @test haskey(ds_h, "bottom_height")
    @test eltype(ds_h[:bottom_height]) == Float64
    @test dimsize(ds_h[:bottom_height]) == (λ_caa=Nλ + 2Hλ, φ_aca=Nφ + 2Hφ)

    for loc in ("ccc", "fcc", "cfc", "ccf")
        @test haskey(ds_h, "immersed_boundary_mask_$loc")
        @test eltype(ds_h["immersed_boundary_mask_$loc"]) == Float64
    end

    @test dimsize(ds_h[:immersed_boundary_mask_ccc]) == (λ_caa=Nλ + 2Hλ,     φ_aca=Nφ + 2Hφ,     z_aac=Nz + 2Hz)
    @test dimsize(ds_h[:immersed_boundary_mask_fcc]) == (λ_faa=Nλ + 2Hλ + 1, φ_aca=Nφ + 2Hφ,     z_aac=Nz + 2Hz)
    @test dimsize(ds_h[:immersed_boundary_mask_cfc]) == (λ_caa=Nλ + 2Hλ,     φ_afa=Nφ + 2Hφ + 1, z_aac=Nz + 2Hz)
    @test dimsize(ds_h[:immersed_boundary_mask_ccf]) == (λ_caa=Nλ + 2Hλ,     φ_aca=Nφ + 2Hφ,     z_aaf=Nz + 2Hz + 1)

    @test all(ds_h[:bottom_height][:, :] .≈ Array(parent(grid.immersed_boundary.bottom_height)))

    close(ds_h)
    rm(filepath_with_halos)

    # Test NetCDF output without halos
    ds_n = NCDataset(filepath_no_halos)

    @test haskey(ds_n, "bottom_height")
    @test eltype(ds_n[:bottom_height]) == Float32
    @test dimsize(ds_n[:bottom_height]) == (λ_caa=Nλ, φ_aca=Nφ)

    for loc in ("ccc", "fcc", "cfc", "ccf")
        @test haskey(ds_n, "immersed_boundary_mask_$loc")
        @test eltype(ds_n["immersed_boundary_mask_$loc"]) == Float32
    end

    @test dimsize(ds_n[:immersed_boundary_mask_ccc]) == (λ_caa=Nλ,     φ_aca=Nφ,     z_aac=Nz)
    @test dimsize(ds_n[:immersed_boundary_mask_fcc]) == (λ_faa=Nλ + 1, φ_aca=Nφ,     z_aac=Nz)
    @test dimsize(ds_n[:immersed_boundary_mask_cfc]) == (λ_caa=Nλ,     φ_afa=Nφ + 1, z_aac=Nz)
    @test dimsize(ds_n[:immersed_boundary_mask_ccf]) == (λ_caa=Nλ,     φ_aca=Nφ,     z_aaf=Nz + 1)

    @test all(ds_n[:bottom_height][:, :] .≈ Array(interior(grid.immersed_boundary.bottom_height)))

    close(ds_n)
    rm(filepath_no_halos)

    # Test NetCDF sliced output
    ds_s = NCDataset(filepath_sliced)

    @test haskey(ds_s, "bottom_height")
    @test eltype(ds_s[:bottom_height]) == Float32
    @test dimsize(ds_s[:bottom_height]) == (λ_caa=nλ, φ_aca=nφ)

    for loc in ("ccc", "fcc", "cfc", "ccf")
        @test haskey(ds_s, "immersed_boundary_mask_$loc")
        @test eltype(ds_s["immersed_boundary_mask_$loc"]) == Float32
    end

    @test dimsize(ds_s[:immersed_boundary_mask_ccc]) == (λ_caa=nλ, φ_aca=nφ, z_aac=nz)
    @test dimsize(ds_s[:immersed_boundary_mask_fcc]) == (λ_faa=nλ, φ_aca=nφ, z_aac=nz)
    @test dimsize(ds_s[:immersed_boundary_mask_cfc]) == (λ_caa=nλ, φ_afa=nφ, z_aac=nz)
    @test dimsize(ds_s[:immersed_boundary_mask_ccf]) == (λ_caa=nλ, φ_aca=nφ, z_aaf=nz)

    @test all(ds_s[:bottom_height][:, :] .≈ Array(interior(grid.immersed_boundary.bottom_height, i_slice, j_slice)))

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
                           extent = (π, 7))

    model = NonhydrostaticModel(; grid,
                                  closure = ScalarDiffusivity(ν=4e-2, κ=4e-2),
                                  buoyancy = SeawaterBuoyancy(),
                                  tracers = (:T, :S))

    Nt = 7
    simulation = Simulation(model, Δt=0.1, stop_iteration=Nt)

    Arch = typeof(arch)
    filepath_with_halos = "test_netcdf_rectilinear_flat_xy_$Arch.nc"
    isfile(filepath_with_halos) && rm(filepath_with_halos)

    simulation.output_writers[:with_halos] = NetCDFWriter(model, fields(model),
                                                          filename = filepath_with_halos,
                                                          schedule = IterationInterval(1),
                                                          array_type = Array{Float64},
                                                          with_halos = true,
                                                          include_grid_metrics = true,
                                                          verbose = true)

    i_slice = 3:6
    j_slice = Ny

    nx = length(i_slice)
    ny = 1

    filepath_sliced = "test_netcdf_rectilinear_flat_xy_sliced_$(Arch).nc"
    isfile(filepath_sliced) && rm(filepath_sliced)

    simulation.output_writers[:sliced] = NetCDFWriter(model, fields(model),
                                                      filename = filepath_sliced,
                                                      indices = (i_slice, j_slice, :),
                                                      schedule = IterationInterval(1),
                                                      array_type = Array{Float64},
                                                      with_halos = false,
                                                      include_grid_metrics = true,
                                                      verbose = true)

    run!(simulation)

    # Test NetCDF output with halos
    ds_h = NCDataset(filepath_with_halos)

    dims = ("x_faa", "x_caa", "y_afa", "y_aca")
    not_dims = ("z_aaf", "z_aac")

    metrics = ("Δx_faa", "Δx_caa", "Δy_afa", "Δy_aca")
    not_metrics = ("Δz_aaf", "Δz_aac")

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

    @test dimsize(ds_h[:x_faa]) == (x_faa=Nx + 2Hx,)
    @test dimsize(ds_h[:x_caa]) == (x_caa=Nx + 2Hx,)
    @test dimsize(ds_h[:y_afa]) == (y_afa=Ny + 2Hy + 1,)
    @test dimsize(ds_h[:y_aca]) == (y_aca=Ny + 2Hy,)

    @test dimsize(ds_h[:Δx_faa]) == (x_faa=Nx + 2Hx,)
    @test dimsize(ds_h[:Δx_caa]) == (x_caa=Nx + 2Hx,)
    @test dimsize(ds_h[:Δy_afa]) == (y_afa=Ny + 2Hy + 1,)
    @test dimsize(ds_h[:Δy_aca]) == (y_aca=Ny + 2Hy,)

    @test dimsize(ds_h[:u]) == (x_faa=Nx + 2Hx, y_aca=Ny + 2Hy,     time=Nt + 1)
    @test dimsize(ds_h[:v]) == (x_caa=Nx + 2Hx, y_afa=Ny + 2Hy + 1, time=Nt + 1)
    @test dimsize(ds_h[:w]) == (x_caa=Nx + 2Hx, y_aca=Ny + 2Hy,     time=Nt + 1)
    @test dimsize(ds_h[:T]) == (x_caa=Nx + 2Hx, y_aca=Ny + 2Hy,     time=Nt + 1)
    @test dimsize(ds_h[:S]) == (x_caa=Nx + 2Hx, y_aca=Ny + 2Hy,     time=Nt + 1)

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

    @test dimsize(ds_s[:x_faa]) == (x_faa=nx,)
    @test dimsize(ds_s[:x_caa]) == (x_caa=nx,)
    @test dimsize(ds_s[:y_afa]) == (y_afa=ny,)
    @test dimsize(ds_s[:y_aca]) == (y_aca=ny,)

    @test dimsize(ds_s[:Δx_faa]) == (x_faa=nx,)
    @test dimsize(ds_s[:Δx_caa]) == (x_caa=nx,)
    @test dimsize(ds_s[:Δy_afa]) == (y_afa=ny,)
    @test dimsize(ds_s[:Δy_aca]) == (y_aca=ny,)

    @test dimsize(ds_s[:u]) == (x_faa=nx, y_aca=ny, time=Nt + 1)
    @test dimsize(ds_s[:v]) == (x_caa=nx, y_afa=ny, time=Nt + 1)
    @test dimsize(ds_s[:w]) == (x_caa=nx, y_aca=ny, time=Nt + 1)
    @test dimsize(ds_s[:T]) == (x_caa=nx, y_aca=ny, time=Nt + 1)
    @test dimsize(ds_s[:S]) == (x_caa=nx, y_aca=ny, time=Nt + 1)

    close(ds_s)
    rm(filepath_sliced)

    return nothing
end

function test_netcdf_rectilinear_flat_xz(arch; immersed)
    Nx, Nz = 8, 8
    Hx, Hz = 2, 3
    Lx, H  = 2, 1

    grid = RectilinearGrid(arch,
                           topology = (Periodic, Flat, Bounded),
                           size = (Nx, Nz),
                           halo = (Hx, Hz),
                           x = (-Lx, Lx),
                           z = (-H, 0))

    if immersed
        height = H / 2
        width = Lx / 3
        mount(x) = height * exp(-x^2 / 2width^2)
        bottom(x) = -H + mount(x)

        grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom))
    end

    model = NonhydrostaticModel(; grid,
                                  closure = ScalarDiffusivity(ν=4e-2, κ=4e-2),
                                  buoyancy = SeawaterBuoyancy(),
                                  tracers = (:T, :S))

    Nt = 7
    simulation = Simulation(model, Δt=0.1, stop_iteration=Nt)

    Arch = typeof(arch)
    filepath_with_halos = "test_netcdf_rectilinear_flat_xz_$Arch.nc"
    isfile(filepath_with_halos) && rm(filepath_with_halos)

    simulation.output_writers[:with_halos] = NetCDFWriter(model, fields(model),
                                                          filename = filepath_with_halos,
                                                          schedule = IterationInterval(1),
                                                          array_type = Array{Float64},
                                                          with_halos = true,
                                                          include_grid_metrics = true,
                                                          verbose = true)

    i_slice = 3:6
    k_slice = Nz

    nx = length(i_slice)
    nz = 1

    filepath_sliced = "test_netcdf_rectilinear_flat_xz_sliced_$(Arch).nc"
    isfile(filepath_sliced) && rm(filepath_sliced)

    simulation.output_writers[:sliced] = NetCDFWriter(model, fields(model),
                                                      filename = filepath_sliced,
                                                      indices = (i_slice, :, k_slice),
                                                      schedule = IterationInterval(1),
                                                      array_type = Array{Float64},
                                                      with_halos = false,
                                                      include_grid_metrics = true,
                                                      verbose = true)

    run!(simulation)

    # Test NetCDF output with halos
    ds_h = NCDataset(filepath_with_halos)

    dims = ("x_faa", "x_caa", "z_aaf", "z_aac")
    not_dims = ("y_afa", "y_aca")

    metrics = ("Δx_faa", "Δx_caa", "Δz_aaf", "Δz_aac")
    not_metrics = ("Δy_afa", "Δy_aca")

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

    @test dimsize(ds_h[:x_faa]) == (x_faa=Nx + 2Hx,)
    @test dimsize(ds_h[:x_caa]) == (x_caa=Nx + 2Hx,)
    @test dimsize(ds_h[:z_aaf]) == (z_aaf=Nz + 2Hz + 1,)
    @test dimsize(ds_h[:z_aac]) == (z_aac=Nz + 2Hz,)

    @test dimsize(ds_h[:Δx_faa]) == (x_faa=Nx + 2Hx,)
    @test dimsize(ds_h[:Δx_caa]) == (x_caa=Nx + 2Hx,)
    @test dimsize(ds_h[:Δz_aaf]) == (z_aaf=Nz + 2Hz + 1,)
    @test dimsize(ds_h[:Δz_aac]) == (z_aac=Nz + 2Hz,)

    @test dimsize(ds_h[:u]) == (x_faa=Nx + 2Hx, z_aac=Nz + 2Hz,     time=Nt + 1)
    @test dimsize(ds_h[:v]) == (x_caa=Nx + 2Hx, z_aac=Nz + 2Hz,     time=Nt + 1)
    @test dimsize(ds_h[:w]) == (x_caa=Nx + 2Hx, z_aaf=Nz + 2Hz + 1, time=Nt + 1)
    @test dimsize(ds_h[:T]) == (x_caa=Nx + 2Hx, z_aac=Nz + 2Hz,     time=Nt + 1)
    @test dimsize(ds_h[:S]) == (x_caa=Nx + 2Hx, z_aac=Nz + 2Hz,     time=Nt + 1)

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

    @test dimsize(ds_s[:x_faa]) == (x_faa=nx,)
    @test dimsize(ds_s[:x_caa]) == (x_caa=nx,)
    @test dimsize(ds_s[:z_aaf]) == (z_aaf=nz,)
    @test dimsize(ds_s[:z_aac]) == (z_aac=nz,)

    @test dimsize(ds_s[:Δx_faa]) == (x_faa=nx,)
    @test dimsize(ds_s[:Δx_caa]) == (x_caa=nx,)
    @test dimsize(ds_s[:Δz_aaf]) == (z_aaf=nz,)
    @test dimsize(ds_s[:Δz_aac]) == (z_aac=nz,)

    @test dimsize(ds_s[:u]) == (x_faa=nx, z_aac=nz, time=Nt + 1)
    @test dimsize(ds_s[:v]) == (x_caa=nx, z_aac=nz, time=Nt + 1)
    @test dimsize(ds_s[:w]) == (x_caa=nx, z_aaf=nz, time=Nt + 1)
    @test dimsize(ds_s[:T]) == (x_caa=nx, z_aac=nz, time=Nt + 1)
    @test dimsize(ds_s[:S]) == (x_caa=nx, z_aac=nz, time=Nt + 1)

    close(ds_s)
    rm(filepath_sliced)

    return nothing
end

function test_netcdf_rectilinear_flat_yz(arch; immersed)
    Ny, Nz = 8, 8
    Hy, Hz = 2, 3
    Ly, H  = 2, 1

    grid = RectilinearGrid(arch,
        topology = (Flat, Periodic, Bounded),
        size = (Ny, Nz),
        halo = (Hy, Hz),
        y = (-Ly, Ly),
        z = (-H, 0)
    )

    if immersed
        height = H / 2
        width = Ly / 3
        mount(x) = height * exp(-x^2 / 2width^2)
        bottom(x) = -H + mount(x)

        grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom))
    end

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
        NetCDFWriter(model, fields(model),
            filename = filepath_with_halos,
            schedule = IterationInterval(1),
            array_type = Array{Float64},
            with_halos = true,
            include_grid_metrics = true,
            verbose = true)

    j_slice = 3:6
    k_slice = Nz

    ny = length(j_slice)
    nz = 1

    filepath_sliced = "test_netcdf_rectilinear_flat_yz_sliced_$(Arch).nc"
    isfile(filepath_sliced) && rm(filepath_sliced)

    simulation.output_writers[:sliced] =
        NetCDFWriter(model, fields(model),
            filename = filepath_sliced,
            indices = (:, j_slice, k_slice),
            schedule = IterationInterval(1),
            array_type = Array{Float64},
            with_halos = false,
            include_grid_metrics = true,
            verbose = true)

    run!(simulation)

    # Test NetCDF output with halos
    ds_h = NCDataset(filepath_with_halos)

    dims = ("y_afa", "y_aca", "z_aaf", "z_aac")
    not_dims = ("x_faa", "x_caa")

    metrics = ("Δy_afa", "Δy_aca", "Δz_aaf", "Δz_aac")
    not_metrics = ("Δx_faa", "Δx_caa")

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

    @test dimsize(ds_h[:y_afa]) == (y_afa=Ny + 2Hy,)
    @test dimsize(ds_h[:y_aca]) == (y_aca=Ny + 2Hy,)
    @test dimsize(ds_h[:z_aaf]) == (z_aaf=Nz + 2Hz + 1,)
    @test dimsize(ds_h[:z_aac]) == (z_aac=Nz + 2Hz,)

    @test dimsize(ds_h[:Δy_afa]) == (y_afa=Ny + 2Hy,)
    @test dimsize(ds_h[:Δy_aca]) == (y_aca=Ny + 2Hy,)
    @test dimsize(ds_h[:Δz_aaf]) == (z_aaf=Nz + 2Hz + 1,)
    @test dimsize(ds_h[:Δz_aac]) == (z_aac=Nz + 2Hz,)

    @test dimsize(ds_h[:u]) == (y_aca=Ny + 2Hy, z_aac=Nz + 2Hz,     time=Nt + 1)
    @test dimsize(ds_h[:v]) == (y_afa=Ny + 2Hy, z_aac=Nz + 2Hz,     time=Nt + 1)
    @test dimsize(ds_h[:w]) == (y_aca=Ny + 2Hy, z_aaf=Nz + 2Hz + 1, time=Nt + 1)
    @test dimsize(ds_h[:T]) == (y_aca=Ny + 2Hy, z_aac=Nz + 2Hz,     time=Nt + 1)
    @test dimsize(ds_h[:S]) == (y_aca=Ny + 2Hy, z_aac=Nz + 2Hz,     time=Nt + 1)

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

    @test dimsize(ds_s[:y_afa]) == (y_afa=ny,)
    @test dimsize(ds_s[:y_aca]) == (y_aca=ny,)
    @test dimsize(ds_s[:z_aaf]) == (z_aaf=nz,)
    @test dimsize(ds_s[:z_aac]) == (z_aac=nz,)

    @test dimsize(ds_s[:Δy_afa]) == (y_afa=ny,)
    @test dimsize(ds_s[:Δy_aca]) == (y_aca=ny,)
    @test dimsize(ds_s[:Δz_aaf]) == (z_aaf=nz,)
    @test dimsize(ds_s[:Δz_aac]) == (z_aac=nz,)

    @test dimsize(ds_s[:u]) == (y_aca=ny, z_aac=nz, time=Nt + 1)
    @test dimsize(ds_s[:v]) == (y_afa=ny, z_aac=nz, time=Nt + 1)
    @test dimsize(ds_s[:w]) == (y_aca=ny, z_aaf=nz, time=Nt + 1)
    @test dimsize(ds_s[:T]) == (y_aca=ny, z_aac=nz, time=Nt + 1)
    @test dimsize(ds_s[:S]) == (y_aca=ny, z_aac=nz, time=Nt + 1)

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
        NetCDFWriter(model, fields(model),
            filename = filepath_with_halos,
            schedule = IterationInterval(1),
            array_type = Array{Float64},
            with_halos = true,
            include_grid_metrics = true,
            verbose = true)

    k_slice = 5:10
    n = length(k_slice)

    filepath_sliced = "test_netcdf_rectilinear_column_sliced_$(Arch).nc"
    isfile(filepath_sliced) && rm(filepath_sliced)

    simulation.output_writers[:sliced] =
        NetCDFWriter(model, fields(model),
            filename = filepath_sliced,
            indices = (:, :, k_slice),
            schedule = IterationInterval(1),
            array_type = Array{Float64},
            with_halos = false,
            include_grid_metrics = true,
            verbose = true)

    run!(simulation)

    # Test NetCDF output with halos
    ds_h = NCDataset(filepath_with_halos)

    dims = ("z_aaf", "z_aac")
    not_dims = ("x_faa", "x_caa", "y_afa", "y_aca")

    metrics = ("Δz_aaf", "Δz_aac")
    not_metrics = ("Δx_faa", "Δx_caa", "Δy_afa", "Δy_aca")

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

    @test dimsize(ds_h[:z_aaf]) == (z_aaf=N + 2H + 1,)
    @test dimsize(ds_h[:z_aac]) == (z_aac=N + 2H,)

    @test dimsize(ds_h[:Δz_aaf]) == (z_aaf=N + 2H + 1,)
    @test dimsize(ds_h[:Δz_aac]) == (z_aac=N + 2H,)

    @test dimsize(ds_h[:u]) == (z_aac=N + 2H,     time=Nt + 1)
    @test dimsize(ds_h[:v]) == (z_aac=N + 2H,     time=Nt + 1)
    @test dimsize(ds_h[:w]) == (z_aaf=N + 2H + 1, time=Nt + 1)
    @test dimsize(ds_h[:T]) == (z_aac=N + 2H,     time=Nt + 1)
    @test dimsize(ds_h[:S]) == (z_aac=N + 2H,     time=Nt + 1)

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

    @test dimsize(ds_s[:z_aaf]) == (z_aaf=n,)
    @test dimsize(ds_s[:z_aac]) == (z_aac=n,)

    @test dimsize(ds_s[:Δz_aaf]) == (z_aaf=n,)
    @test dimsize(ds_s[:Δz_aac]) == (z_aac=n,)

    @test dimsize(ds_s[:u]) == (z_aac=n, time=Nt + 1)
    @test dimsize(ds_s[:v]) == (z_aac=n, time=Nt + 1)
    @test dimsize(ds_s[:w]) == (z_aaf=n, time=Nt + 1)
    @test dimsize(ds_s[:T]) == (z_aac=n, time=Nt + 1)
    @test dimsize(ds_s[:S]) == (z_aac=n, time=Nt + 1)

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

    nc_writer = NetCDFWriter(model, outputs,
        filename = nc_filepath,
        schedule = IterationInterval(10),
        array_type = Array{FT},
        include_grid_metrics = false,
        verbose = true)

    push!(simulation.output_writers, nc_writer)

    i_slice = 1:10
    j_slice = 13
    k_slice = 9:11
    indices = (i_slice, j_slice, k_slice)
    j_slice = j_slice:j_slice  # So we can correctly index with it for later tests.

    nc_sliced_filepath = "test_thermal_bubble_sliced_$(Arch)_$FT.nc"
    isfile(nc_sliced_filepath) && rm(nc_sliced_filepath)

    nc_sliced_writer = NetCDFWriter(model, outputs,
        filename = nc_sliced_filepath,
        schedule = IterationInterval(10),
        array_type = Array{FT},
        indices = indices,
        include_grid_metrics = false,
        verbose = true)

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

    @test eltype(ds3["x_caa"]) == FT
    @test eltype(ds3["x_faa"]) == FT
    @test eltype(ds3["y_aca"]) == FT
    @test eltype(ds3["y_afa"]) == FT
    @test eltype(ds3["z_aac"]) == FT
    @test eltype(ds3["z_aaf"]) == FT

    @test length(ds3["x_caa"]) == Nx
    @test length(ds3["y_aca"]) == Ny
    @test length(ds3["z_aac"]) == Nz
    @test length(ds3["x_faa"]) == Nx
    @test length(ds3["y_afa"]) == Ny
    @test length(ds3["z_aaf"]) == Nz+1  # z is Bounded

    @test ds3["x_caa"][1] == grid.xᶜᵃᵃ[1]
    @test ds3["x_faa"][1] == grid.xᶠᵃᵃ[1]
    @test ds3["y_aca"][1] == grid.yᵃᶜᵃ[1]
    @test ds3["y_afa"][1] == grid.yᵃᶠᵃ[1]
    @test ds3["z_aac"][1] == grid.z.cᵃᵃᶜ[1]
    @test ds3["z_aaf"][1] == grid.z.cᵃᵃᶠ[1]

    @test ds3["x_caa"][end] == grid.xᶜᵃᵃ[Nx]
    @test ds3["x_faa"][end] == grid.xᶠᵃᵃ[Nx]
    @test ds3["y_aca"][end] == grid.yᵃᶜᵃ[Ny]
    @test ds3["y_afa"][end] == grid.yᵃᶠᵃ[Ny]
    @test ds3["z_aac"][end] == grid.z.cᵃᵃᶜ[Nz]
    @test ds3["z_aaf"][end] == grid.z.cᵃᵃᶠ[Nz+1]  # z is Bounded

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

    @test eltype(ds2["x_caa"]) == FT
    @test eltype(ds2["x_faa"]) == FT
    @test eltype(ds2["y_aca"]) == FT
    @test eltype(ds2["y_afa"]) == FT
    @test eltype(ds2["z_aac"]) == FT
    @test eltype(ds2["z_aaf"]) == FT

    @test length(ds2["x_caa"]) == length(i_slice)
    @test length(ds2["x_faa"]) == length(i_slice)
    @test length(ds2["y_aca"]) == length(j_slice)
    @test length(ds2["y_afa"]) == length(j_slice)
    @test length(ds2["z_aac"]) == length(k_slice)
    @test length(ds2["z_aaf"]) == length(k_slice)

    @test ds2["x_caa"][1] == grid.xᶜᵃᵃ[i_slice[1]]
    @test ds2["x_faa"][1] == grid.xᶠᵃᵃ[i_slice[1]]
    @test ds2["y_aca"][1] == grid.yᵃᶜᵃ[j_slice[1]]
    @test ds2["y_afa"][1] == grid.yᵃᶠᵃ[j_slice[1]]
    @test ds2["z_aac"][1] == grid.z.cᵃᵃᶜ[k_slice[1]]
    @test ds2["z_aaf"][1] == grid.z.cᵃᵃᶠ[k_slice[1]]

    @test ds2["x_caa"][end] == grid.xᶜᵃᵃ[i_slice[end]]
    @test ds2["x_faa"][end] == grid.xᶠᵃᵃ[i_slice[end]]
    @test ds2["y_aca"][end] == grid.yᵃᶜᵃ[j_slice[end]]
    @test ds2["y_afa"][end] == grid.yᵃᶠᵃ[j_slice[end]]
    @test ds2["z_aac"][end] == grid.z.cᵃᵃᶜ[k_slice[end]]
    @test ds2["z_aaf"][end] == grid.z.cᵃᵃᶠ[k_slice[end]]

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

    nc_writer = NetCDFWriter(model,
        merge(model.velocities, model.tracers),
        filename = nc_filepath,
        schedule = IterationInterval(10),
        array_type = Array{FT},
        with_halos = true)

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

    @test eltype(ds["x_caa"]) == FT
    @test eltype(ds["x_faa"]) == FT
    @test eltype(ds["y_aca"]) == FT
    @test eltype(ds["y_afa"]) == FT
    @test eltype(ds["z_aac"]) == FT
    @test eltype(ds["z_aaf"]) == FT

    @test length(ds["x_caa"]) == Nx+2Hx
    @test length(ds["y_aca"]) == Ny+2Hy
    @test length(ds["z_aac"]) == Nz+2Hz
    @test length(ds["x_faa"]) == Nx+2Hx
    @test length(ds["y_afa"]) == Ny+2Hy
    @test length(ds["z_aaf"]) == Nz+2Hz+1  # z is Bounded

    @test ds["x_caa"][1] == grid.xᶜᵃᵃ[1-Hx]
    @test ds["x_faa"][1] == grid.xᶠᵃᵃ[1-Hx]
    @test ds["y_aca"][1] == grid.yᵃᶜᵃ[1-Hy]
    @test ds["y_afa"][1] == grid.yᵃᶠᵃ[1-Hy]
    @test ds["z_aac"][1] == grid.z.cᵃᵃᶜ[1-Hz]
    @test ds["z_aaf"][1] == grid.z.cᵃᵃᶠ[1-Hz]

    @test ds["x_caa"][end] == grid.xᶜᵃᵃ[Nx+Hx]
    @test ds["x_faa"][end] == grid.xᶠᵃᵃ[Nx+Hx]
    @test ds["y_aca"][end] == grid.yᵃᶜᵃ[Ny+Hy]
    @test ds["y_afa"][end] == grid.yᵃᶠᵃ[Ny+Hy]
    @test ds["z_aac"][end] == grid.z.cᵃᵃᶜ[Nz+Hz]
    @test ds["z_aaf"][end] == grid.z.cᵃᵃᶠ[Nz+Hz+1]  # z is Bounded

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
    ow = NetCDFWriter(model, (; u=model.velocities.u);
        dir = ".",
        filename = "test_size_file_splitting_$Arch",
        schedule = IterationInterval(1),
        array_type = Array{Float64},
        with_halos = true,
        global_attributes = fake_attributes,
        file_splitting = FileSizeLimit(200KiB),
        overwrite_existing = true)

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
    ow = NetCDFWriter(model, (; u=model.velocities.u);
        dir = ".",
        filename = "test_time_file_splitting_$Arch",
        schedule = IterationInterval(2),
        array_type = Array{Float64},
        with_halos = true,
        global_attributes = fake_attributes,
        file_splitting = TimeInterval(4seconds),
        overwrite_existing = true)

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

    # Define scalar, vector, and 2D slice outputs
    f(model) = model.clock.time^2

    g(model) = model.clock.time .* exp.(znodes(grid, Center()))

    xC = xnodes(grid, Center())
    yF = ynodes(grid, Face())

    XC = [xC[i] for i in 1:Nx, j in 1:Ny]
    YF = [yF[j] for i in 1:Nx, j in 1:Ny]

    h(model) = @. model.clock.time * sin(XC) * cos(YF) # xy slice output

    outputs = (scalar=f, profile=g, slice=h)
    dims = (scalar=(), profile=("z_aac",), slice=("x_caa", "y_aca"))

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
        NetCDFWriter(model, outputs;
            global_attributes,
            output_attributes,
            filename = nc_filepath,
            schedule = TimeInterval(Δt),
            dimensions = dims,
            array_type = Array{Float64},
            include_grid_metrics = false,
            verbose = true)

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

    @test eltype(ds["x_caa"]) == Float64
    @test eltype(ds["x_faa"]) == Float64
    @test eltype(ds["y_aca"]) == Float64
    @test eltype(ds["y_afa"]) == Float64
    @test eltype(ds["z_aac"]) == Float64
    @test eltype(ds["z_aaf"]) == Float64

    @test length(ds["x_caa"]) == N
    @test length(ds["y_aca"]) == N
    @test length(ds["z_aac"]) == N
    @test length(ds["x_faa"]) == N
    @test length(ds["y_afa"]) == N
    @test length(ds["z_aaf"]) == N+1  # z is Bounded

    @test ds["x_caa"][1] == grid.xᶜᵃᵃ[1]
    @test ds["x_faa"][1] == grid.xᶠᵃᵃ[1]
    @test ds["y_aca"][1] == grid.yᵃᶜᵃ[1]
    @test ds["y_afa"][1] == grid.yᵃᶠᵃ[1]
    @test ds["z_aac"][1] == grid.z.cᵃᵃᶜ[1]
    @test ds["z_aaf"][1] == grid.z.cᵃᵃᶠ[1]

    @test ds["x_caa"][end] == grid.xᶜᵃᵃ[N]
    @test ds["y_aca"][end] == grid.yᵃᶜᵃ[N]
    @test ds["x_faa"][end] == grid.xᶠᵃᵃ[N]
    @test ds["y_afa"][end] == grid.yᵃᶠᵃ[N]
    @test ds["z_aac"][end] == grid.z.cᵃᵃᶜ[N]
    @test ds["z_aaf"][end] == grid.z.cᵃᵃᶠ[N+1]  # z is Bounded

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
    @test dimnames(ds["profile"]) == ("z_aac", "time")

    for n in 0:iters
        @test ds["profile"][:, n+1] == n*Δt .* exp.(znodes(grid, Center()))
    end

    @test ds["slice"].attrib["long_name"] == "Some slice"
    @test ds["slice"].attrib["units"] == "mushrooms"
    @test size(ds["slice"]) == (N, N, iters+1)
    @test dimnames(ds["slice"]) == ("x_caa", "y_aca", "time")

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
        NetCDFWriter(model, outputs;
            global_attributes,
            output_attributes,
            filename = nc_filepath,
            overwrite_existing = false,
            schedule = IterationInterval(1),
            array_type = Array{Float64},
            dimensions = dims,
            verbose = true)

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
        NetCDFWriter(model,
            (; ∫c_dx, ∫∫c_dxdy, ∫∫∫c_dxdydz),
            array_type = Array{Float64},
            verbose = true,
            filename = nc_filepath,
            schedule = IterationInterval(2),
            include_grid_metrics = false)

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
                NetCDFWriter(
                    model,
                    nc_outputs,
                    array_type = Array{Float64},
                    verbose = true,
                    filename = horizontal_average_nc_filepath,
                    schedule = TimeInterval(10Δt),
                    include_grid_metrics = false)

            multiple_time_average_nc_filepath = "decay_windowed_time_average_test_$Arch.nc"
            single_time_average_nc_filepath = "single_decay_windowed_time_average_test_$Arch.nc"
            window = 6Δt

            single_nc_output = Dict("c1" => ∫c1_dxdy)

            simulation.output_writers[:single_output_time_average] =
                NetCDFWriter(
                    model,
                    single_nc_output,
                    array_type = Array{Float64},
                    verbose = true,
                    filename = single_time_average_nc_filepath,
                    schedule = AveragedTimeInterval(10Δt; window, stride),
                    include_grid_metrics = false)

            simulation.output_writers[:multiple_output_time_average] =
                NetCDFWriter(
                    model,
                    nc_outputs,
                    array_type = Array{Float64},
                    verbose = true,
                    filename = multiple_time_average_nc_filepath,
                    schedule = AveragedTimeInterval(10Δt; window, stride),
                    include_grid_metrics = false)

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
        NetCDFWriter(model,
            model.velocities,
            filename = test_filename1,
            schedule = TimeInterval(7.3),
            include_grid_metrics = false)

    test_filename2 = "test_output_alignment2_$Arch.nc"
    simulation.output_writers[:something] =
        NetCDFWriter(model,
            model.tracers,
            filename = test_filename2,
            schedule = TimeInterval(3.0),
            include_grid_metrics = false)

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
        NetCDFWriter(model,
            (; particles = model.particles),
            filename = filepath,
            schedule = IterationInterval(1),
            include_grid_metrics = false)

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
        NetCDFWriter(model, outputs,
            filename = filepath,
            schedule = IterationInterval(1),
            include_grid_metrics = false)

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
        NetCDFWriter(model,
            merge(model.velocities, model.tracers),
            filename = nc_filepath,
            schedule = IterationInterval(1),
            array_type = Array{Float64},
            include_grid_metrics = false,
            verbose = true)

    run!(simulation)

    ds = NCDataset(nc_filepath)

    @test length(ds["x_caa"]) == Nx
    @test length(ds["y_aca"]) == Ny
    @test length(ds["z_aac"]) == Nz
    @test length(ds["x_faa"]) == Nx
    @test length(ds["y_afa"]) == Ny
    @test length(ds["z_aaf"]) == Nz+1  # z is Bounded

    @test ds["x_caa"][1] == grid.xᶜᵃᵃ[1]
    @test ds["x_faa"][1] == grid.xᶠᵃᵃ[1]
    @test ds["y_aca"][1] == grid.yᵃᶜᵃ[1]
    @test ds["y_afa"][1] == grid.yᵃᶠᵃ[1]

    @test CUDA.@allowscalar ds["z_aac"][1] == grid.z.cᵃᵃᶜ[1]
    @test CUDA.@allowscalar ds["z_aaf"][1] == grid.z.cᵃᵃᶠ[1]

    @test ds["x_caa"][end] == grid.xᶜᵃᵃ[Nx]
    @test ds["x_faa"][end] == grid.xᶠᵃᵃ[Nx]
    @test ds["y_aca"][end] == grid.yᵃᶜᵃ[Ny]
    @test ds["y_afa"][end] == grid.yᵃᶠᵃ[Ny]

    @test CUDA.@allowscalar ds["z_aac"][end] == grid.z.cᵃᵃᶜ[Nz]
    @test CUDA.@allowscalar ds["z_aaf"][end] == grid.z.cᵃᵃᶠ[Nz+1]  # z is Bounded

    close(ds)
    rm(nc_filepath)

    return nothing
end

function test_netcdf_overriding_attributes(arch)
    arch = CPU()

    grid = LatitudeLongitudeGrid(arch;
        topology = (Bounded, Bounded, Bounded),
        size = (4, 4, 4),
        longitude = (-1, 1),
        latitude = (-1, 1),
        z = (-100, 0)
    )

    model = HydrostaticFreeSurfaceModel(; grid,
        closure = ScalarDiffusivity(ν=4e-2, κ=4e-2),
        buoyancy = SeawaterBuoyancy(),
        tracers = (:T, :S)
    )

    outputs = merge(model.velocities, model.tracers)

    Arch = typeof(arch)
    nc_filepath = "test_attributes_$Arch.nc"
    isfile(nc_filepath) && rm(nc_filepath)

    global_attributes = Dict(
        "date" => "yesterday", # This should override the default date attribute.
        "fruit" => "papaya"    # This should show up in the NetCDF file.
    )

    output_attributes = Dict(
        "φ_afa" => Dict("units" => "No units for you!"),
        "u" => Dict("long_name" => "zonal velocity", "units" => "miles/fortnight")
    )

    nc_writer = NetCDFWriter(model, outputs;
        filename = nc_filepath,
        schedule = IterationInterval(1),
        global_attributes,
        output_attributes)

    ds = NCDataset(nc_filepath)

    @test ds.attrib["date"] == "yesterday"
    @test ds.attrib["fruit"] == "papaya"

    @test ds["φ_afa"].attrib["units"] == "No units for you!"
    @test !haskey(ds["φ_afa"].attrib, "long_name")

    @test ds["u"].attrib["long_name"] == "zonal velocity"
    @test ds["u"].attrib["units"] == "miles/fortnight"

    return nothing
end

function test_netcdf_free_surface_only_output(arch)
    Nλ, Nφ, Nz = 8, 8, 4
    Hλ, Hφ, Hz = 3, 4, 2

    grid = LatitudeLongitudeGrid(arch;
        topology = (Bounded, Bounded, Bounded),
        size = (Nλ, Nφ, Nz),
        halo = (Hλ, Hφ, Hz),
        longitude = (-1, 1),
        latitude = (-1, 1),
        z = (-100, 0)
    )

    model = HydrostaticFreeSurfaceModel(; grid,
        closure = ScalarDiffusivity(ν=4e-2, κ=4e-2),
        buoyancy = SeawaterBuoyancy(),
        tracers = (:T, :S)
    )

    Nt = 5
    simulation = Simulation(model, Δt=0.1, stop_iteration=Nt)

    # Kind of a hack because we want η to be a ReducedField.
    outputs = (;
        η = Average(model.free_surface.η, dims=3),
    )

    Arch = typeof(arch)
    filepath_with_halos = "test_free_surface_with_halos_$Arch.nc"
    isfile(filepath_with_halos) && rm(filepath_with_halos)

    simulation.output_writers[:with_halos] =
        NetCDFWriter(model, outputs;
            filename = filepath_with_halos,
            schedule = IterationInterval(1),
            with_halos = true)

    filepath_no_halos = "test_free_surface_no_halos_$Arch.nc"
    isfile(filepath_no_halos) && rm(filepath_no_halos)

    simulation.output_writers[:no_halos] =
        NetCDFWriter(model, outputs;
            filename = filepath_no_halos,
            schedule = IterationInterval(1),
            with_halos = false)

    run!(simulation)

    ds_h = NCDataset(filepath_with_halos)

    @test haskey(ds_h, "η")
    @test dimsize(ds_h["η"]) == (λ_caa=Nλ + 2Hλ, φ_aca=Nφ + 2Hφ, time=Nt + 1)

    close(ds_h)
    rm(filepath_with_halos)

    ds_n = NCDataset(filepath_no_halos)

    @test haskey(ds_n, "η")
    @test dimsize(ds_n["η"]) == (λ_caa=Nλ, φ_aca=Nφ, time=Nt + 1)

    close(ds_n)
    rm(filepath_no_halos)

    return nothing
end

function test_netcdf_free_surface_mixed_output(arch)
    Nλ, Nφ, Nz = 8, 8, 4
    Hλ, Hφ, Hz = 3, 4, 2

    grid = LatitudeLongitudeGrid(arch;
        topology = (Bounded, Bounded, Bounded),
        size = (Nλ, Nφ, Nz),
        halo = (Hλ, Hφ, Hz),
        longitude = (-1, 1),
        latitude = (-1, 1),
        z = (-100, 0)
    )

    model = HydrostaticFreeSurfaceModel(; grid,
        closure = ScalarDiffusivity(ν=4e-2, κ=4e-2),
        buoyancy = SeawaterBuoyancy(),
        tracers = (:T, :S)
    )

    Nt = 5
    simulation = Simulation(model, Δt=0.1, stop_iteration=Nt)

    # Kind of a hack because we want η to be a ReducedField.
    free_surface_outputs = (;
        η = Average(model.free_surface.η, dims=3),
    )

    outputs = merge(model.velocities, model.tracers, free_surface_outputs)

    Arch = typeof(arch)
    filepath_with_halos = "test_mixed_free_surface_with_halos_$Arch.nc"
    isfile(filepath_with_halos) && rm(filepath_with_halos)

    simulation.output_writers[:with_halos] =
        NetCDFWriter(model, outputs;
            filename = filepath_with_halos,
            schedule = IterationInterval(1),
            with_halos = true)

    filepath_no_halos = "test_mixed_free_surface_no_halos_$Arch.nc"
    isfile(filepath_no_halos) && rm(filepath_no_halos)

    simulation.output_writers[:no_halos] =
        NetCDFWriter(model, outputs;
            filename = filepath_no_halos,
            schedule = IterationInterval(1),
            with_halos = false)

    run!(simulation)

    ds_h = NCDataset(filepath_with_halos)

    @test haskey(ds_h, "η")
    @test dimsize(ds_h["η"]) == (λ_caa=Nλ + 2Hλ, φ_aca=Nφ + 2Hφ, time=Nt + 1)

    @test dimsize(ds_h[:u]) == (λ_faa=Nλ + 2Hλ + 1, φ_aca=Nφ + 2Hφ,     z_aac=Nz + 2Hz,     time=Nt + 1)
    @test dimsize(ds_h[:v]) == (λ_caa=Nλ + 2Hλ,     φ_afa=Nφ + 2Hφ + 1, z_aac=Nz + 2Hz,     time=Nt + 1)
    @test dimsize(ds_h[:w]) == (λ_caa=Nλ + 2Hλ,     φ_aca=Nφ + 2Hφ,     z_aaf=Nz + 2Hz + 1, time=Nt + 1)
    @test dimsize(ds_h[:T]) == (λ_caa=Nλ + 2Hλ,     φ_aca=Nφ + 2Hφ,     z_aac=Nz + 2Hz,     time=Nt + 1)
    @test dimsize(ds_h[:S]) == (λ_caa=Nλ + 2Hλ,     φ_aca=Nφ + 2Hφ,     z_aac=Nz + 2Hz,     time=Nt + 1)

    close(ds_h)
    rm(filepath_with_halos)

    ds_n = NCDataset(filepath_no_halos)

    @test haskey(ds_n, "η")
    @test dimsize(ds_n["η"]) == (λ_caa=Nλ, φ_aca=Nφ, time=Nt + 1)

    @test dimsize(ds_n[:u]) == (λ_faa=Nλ + 1, φ_aca=Nφ,     z_aac=Nz,     time=Nt + 1)
    @test dimsize(ds_n[:v]) == (λ_caa=Nλ,     φ_afa=Nφ + 1, z_aac=Nz,     time=Nt + 1)
    @test dimsize(ds_n[:w]) == (λ_caa=Nλ,     φ_aca=Nφ,     z_aaf=Nz + 1, time=Nt + 1)
    @test dimsize(ds_n[:T]) == (λ_caa=Nλ,     φ_aca=Nφ,     z_aac=Nz,     time=Nt + 1)
    @test dimsize(ds_n[:S]) == (λ_caa=Nλ,     φ_aca=Nφ,     z_aac=Nz,     time=Nt + 1)

    close(ds_n)
    rm(filepath_no_halos)

    return nothing
end

function test_netcdf_buoyancy_force(arch)

    Nx, Nz = 8, 8
    Hx, Hz = 2, 3
    Lx, H  = 2, 1

    grid = RectilinearGrid(arch,
                           topology = (Periodic, Flat, Bounded),
                           size = (Nx, Nz),
                           halo = (Hx, Hz),
                           x = (-Lx, Lx),
                           z = (-H, 0))

    Boussinesq_eos = (TEOS10EquationOfState(),
                      RoquetEquationOfState(:Linear),
                      RoquetEquationOfState(:Cabbeling),
                      RoquetEquationOfState(:CabbelingThermobaricity),
                      RoquetEquationOfState(:Freezing),
                      RoquetEquationOfState(:SecondOrder),
                      RoquetEquationOfState(:SimplestRealistic))

    for eos in Boussinesq_eos

        model = NonhydrostaticModel(; grid,
                                    closure = ScalarDiffusivity(ν=4e-2, κ=4e-2),
                                    buoyancy = SeawaterBuoyancy(equation_of_state=eos),
                                    tracers = (:T, :S))

        Nt = 7
        simulation = Simulation(model, Δt=0.1, stop_iteration=Nt)

        simulation.output_writers[:b_eos] = NetCDFWriter(model, fields(model),
                                                         filename = string(eos)*"_.nc",
                                                         schedule = IterationInterval(1),
                                                         array_type = Array{Float64},
                                                         include_grid_metrics = true,
                                                         verbose = true)
        # only tests that the writer builds, produces a file at filepath and sets attributes
        @test simulation.output_writers[:b_eos] isa NetCDFWriter
        @test isfile(simulation.output_writers[:b_eos].filepath)
        ds = NCDataset(simulation.output_writers[:b_eos].filepath)
        @test ds["T"].attrib["long_name"] == "Conservative temperature"
        @test ds["T"].attrib["units"] == "°C"
        @test ds["S"].attrib["long_name"] == "Absolute salinity"
        @test ds["S"].attrib["units"] == "g/kg"
        close(ds)
        rm(simulation.output_writers[:b_eos].filepath)
    end
    return nothing
end

for arch in archs
    @testset "NetCDF output writer [$(typeof(arch))]" begin
        @info "  Testing NetCDF output writer [$(typeof(arch))]..."

        test_datetime_netcdf_output(arch)
        test_timedate_netcdf_output(arch)

        test_netcdf_grid_metrics_rectilinear(arch, Float64)
        test_netcdf_grid_metrics_rectilinear(arch, Float32)
        test_netcdf_grid_metrics_latlon(arch, Float64)
        test_netcdf_grid_metrics_latlon(arch, Float32)

        test_netcdf_rectilinear_grid_fitted_bottom(arch)
        test_netcdf_latlon_grid_fitted_bottom(arch)

        test_netcdf_rectilinear_flat_xy(arch)
        test_netcdf_rectilinear_flat_xz(arch, immersed=false)
        test_netcdf_rectilinear_flat_xz(arch, immersed=true)
        test_netcdf_rectilinear_flat_yz(arch, immersed=false)
        test_netcdf_rectilinear_flat_yz(arch, immersed=true)
        test_netcdf_rectilinear_column(arch)

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

        test_netcdf_overriding_attributes(arch)

        test_netcdf_free_surface_only_output(arch)
        test_netcdf_free_surface_mixed_output(arch)

        test_netcdf_buoyancy_force(arch)
    end
end
