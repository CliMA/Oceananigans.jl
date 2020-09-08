using Statistics
using NCDatasets
using Oceananigans.BoundaryConditions: BoundaryFunction, PBC, FBC, ZFBC
using Oceananigans.Diagnostics

function run_thermal_bubble_netcdf_tests(arch)
    Nx, Ny, Nz = 16, 16, 16
    Lx, Ly, Lz = 100, 100, 100

    grid = RegularCartesianGrid(size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz))
    closure = IsotropicDiffusivity(ν=4e-2, κ=4e-2)
    model = IncompressibleModel(architecture=arch, grid=grid, closure=closure)
    simulation = Simulation(model, Δt=6, stop_iteration=10)

    # Add a cube-shaped warm temperature anomaly that takes up the middle 50%
    # of the domain volume.
    i1, i2 = round(Int, Nx/4), round(Int, 3Nx/4)
    j1, j2 = round(Int, Ny/4), round(Int, 3Ny/4)
    k1, k2 = round(Int, Nz/4), round(Int, 3Nz/4)
    CUDA.@allowscalar model.tracers.T.data[i1:i2, j1:j2, k1:k2] .+= 0.01

    outputs = Dict(
        "v" => model.velocities.v,
        "u" => model.velocities.u,
        "w" => model.velocities.w,
        "T" => model.tracers.T,
        "S" => model.tracers.S
    )
    nc_filename = "test_dump_$(typeof(arch)).nc"
    nc_writer = NetCDFOutputWriter(model, outputs, filename=nc_filename, iteration_interval=10, verbose=true)
    push!(simulation.output_writers, nc_writer)

    xC_slice = 1:10
    xF_slice = 2:11
    yC_slice = 10:15
    yF_slice = 1
    zC_slice = 10
    zF_slice = 9:11

    nc_sliced_filename = "test_dump_sliced_$(typeof(arch)).nc"
    nc_sliced_writer =
        NetCDFOutputWriter(model, outputs, filename=nc_sliced_filename, iteration_interval=10, verbose=true,
                           xC=xC_slice, xF=xF_slice, yC=yC_slice,
                           yF=yF_slice, zC=zC_slice, zF=zF_slice)

    push!(simulation.output_writers, nc_sliced_writer)

    run!(simulation)

    ds3 = Dataset(nc_filename)

    @test haskey(ds3.attrib, "date") && !isnothing(ds3.attrib["date"])
    @test haskey(ds3.attrib, "Julia") && !isnothing(ds3.attrib["Julia"])
    @test haskey(ds3.attrib, "Oceananigans") && !isnothing(ds3.attrib["Oceananigans"])

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

    ds2 = Dataset(nc_sliced_filename)

    @test haskey(ds2.attrib, "date") && !isnothing(ds2.attrib["date"])
    @test haskey(ds2.attrib, "Julia") && !isnothing(ds2.attrib["Julia"])
    @test haskey(ds2.attrib, "Oceananigans") && !isnothing(ds2.attrib["Oceananigans"])

    @test length(ds2["xC"]) == length(xC_slice)
    @test length(ds2["xF"]) == length(xF_slice)
    @test length(ds2["yC"]) == length(yC_slice)
    @test length(ds2["yF"]) == length(yF_slice)
    @test length(ds2["zC"]) == length(zC_slice)
    @test length(ds2["zF"]) == length(zF_slice)

    @test ds2["xC"][1] == grid.xC[xC_slice[1]]
    @test ds2["xF"][1] == grid.xF[xF_slice[1]]
    @test ds2["yC"][1] == grid.yC[yC_slice[1]]
    @test ds2["yF"][1] == grid.yF[yF_slice]
    @test ds2["zC"][1] == grid.zC[zC_slice]
    @test ds2["zF"][1] == grid.zF[zF_slice[1]]

    @test ds2["xC"][end] == grid.xC[xC_slice[end]]
    @test ds2["xF"][end] == grid.xF[xF_slice[end]]
    @test ds2["yC"][end] == grid.yC[yC_slice[end]]
    @test ds2["yF"][end] == grid.yF[yF_slice]
    @test ds2["zC"][end] == grid.zC[zC_slice]
    @test ds2["zF"][end] == grid.zF[zF_slice[end]]

    u_sliced = ds2["u"][:, :, :, end]
    v_sliced = ds2["v"][:, :, :, end]
    w_sliced = ds2["w"][:, :, :, end]
    T_sliced = ds2["T"][:, :, :, end]
    S_sliced = ds2["S"][:, :, :, end]

    close(ds2)

    @test all(u_sliced .≈ Array(interiorparent(model.velocities.u))[xF_slice, yC_slice, zC_slice])
    @test all(v_sliced .≈ Array(interiorparent(model.velocities.v))[xC_slice, yF_slice, zC_slice])
    @test all(w_sliced .≈ Array(interiorparent(model.velocities.w))[xC_slice, yC_slice, zF_slice])
    @test all(T_sliced .≈ Array(interiorparent(model.tracers.T))[xC_slice, yC_slice, zC_slice])
    @test all(S_sliced .≈ Array(interiorparent(model.tracers.S))[xC_slice, yC_slice, zC_slice])
end

function run_thermal_bubble_netcdf_tests_with_halos(arch)
    Nx, Ny, Nz = 16, 16, 16
    Lx, Ly, Lz = 100, 100, 100

    grid = RegularCartesianGrid(size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz))
    closure = IsotropicDiffusivity(ν=4e-2, κ=4e-2)
    model = IncompressibleModel(architecture=arch, grid=grid, closure=closure)
    simulation = Simulation(model, Δt=6, stop_iteration=10)

    # Add a cube-shaped warm temperature anomaly that takes up the middle 50%
    # of the domain volume.
    i1, i2 = round(Int, Nx/4), round(Int, 3Nx/4)
    j1, j2 = round(Int, Ny/4), round(Int, 3Ny/4)
    k1, k2 = round(Int, Nz/4), round(Int, 3Nz/4)
    CUDA.@allowscalar model.tracers.T.data[i1:i2, j1:j2, k1:k2] .+= 0.01

    outputs = Dict(
        "v" => model.velocities.v,
        "u" => model.velocities.u,
        "w" => model.velocities.w,
        "T" => model.tracers.T,
        "S" => model.tracers.S
    )
    nc_filename = "test_dump_with_halos_$(typeof(arch)).nc"
    nc_writer = NetCDFOutputWriter(model, outputs, filename=nc_filename, iteration_interval=10, with_halos=true)
    push!(simulation.output_writers, nc_writer)

    run!(simulation)

    ds = Dataset(nc_filename)

    @test haskey(ds.attrib, "date") && !isnothing(ds.attrib["date"])
    @test haskey(ds.attrib, "Julia") && !isnothing(ds.attrib["Julia"])
    @test haskey(ds.attrib, "Oceananigans") && !isnothing(ds.attrib["Oceananigans"])

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
end

function run_netcdf_function_output_tests(arch)
    N = 16
    L = 1
    Δt = 1.25
    iters = 3

    model = IncompressibleModel(grid=RegularCartesianGrid(size=(N, N, N), extent=(L, 2L, 3L)))
    simulation = Simulation(model, Δt=Δt, stop_iteration=iters)
    grid = model.grid

    # Define scalar, vector, and 2D slice outputs
    f(model) = model.clock.time^2

    g(model) = model.clock.time .* exp.(znodes(Cell, grid))

    h(model) = model.clock.time .* (   sin.(xnodes(Cell, grid, reshape=true)[:, :, 1])
                                    .* cos.(ynodes(Face, grid, reshape=true)[:, :, 1]))

    outputs = Dict("scalar" => f,  "profile" => g,       "slice" => h)
       dims = Dict("scalar" => (), "profile" => ("zC",), "slice" => ("xC", "yC"))

    output_attributes = Dict(
        "scalar"  => Dict("longname" => "Some scalar", "units" => "bananas"),
        "profile" => Dict("longname" => "Some vertical profile", "units" => "watermelons"),
        "slice"   => Dict("longname" => "Some slice", "units" => "mushrooms")
    )

    global_attributes = Dict("location" => "Bay of Fundy", "onions" => 7)

    nc_filename = "test_function_outputs_$(typeof(arch)).nc"
    simulation.output_writers[:food] =
        NetCDFOutputWriter(model, outputs;
            iteration_interval=1, filename=nc_filename, dimensions=dims, verbose=true,
            global_attributes=global_attributes, output_attributes=output_attributes)

    run!(simulation)

    ds = Dataset(nc_filename, "r")

    @test haskey(ds.attrib, "date") && !isnothing(ds.attrib["date"])
    @test haskey(ds.attrib, "Julia") && !isnothing(ds.attrib["Julia"])
    @test haskey(ds.attrib, "Oceananigans") && !isnothing(ds.attrib["Oceananigans"])

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
        @test ds["profile"][:, n+1] == n*Δt .* exp.(znodes(Cell, grid))
    end

    @test ds["slice"].attrib["longname"] == "Some slice"
    @test ds["slice"].attrib["units"] == "mushrooms"
    @test size(ds["slice"]) == (N, N, iters+1)
    @test dimnames(ds["slice"]) == ("xC", "yC", "time")

    for n in 0:iters
        @test ds["slice"][:, :, n+1] == n*Δt .* (   sin.(xnodes(Cell, grid, reshape=true)[:, :, 1])
                                                 .* cos.(ynodes(Face, grid, reshape=true)[:, :, 1]))
    end

    close(simulation.output_writers[:food])

    #####
    ##### Take 1 more time step and test that appending to a NetCDF file works
    #####

    iters += 1
    simulation = Simulation(model, Δt=Δt, stop_iteration=iters)

    simulation.output_writers[:food] =
        NetCDFOutputWriter(model, outputs;
            iteration_interval=1, filename=nc_filename, mode="a", dimensions=dims, verbose=true,
            global_attributes=global_attributes, output_attributes=output_attributes)

    run!(simulation)

    ds = Dataset(nc_filename, "r")

    @test length(ds["time"]) == iters+1
    @test length(ds["scalar"]) == iters+1
    @test size(ds["profile"]) == (N, iters+1)
    @test size(ds["slice"]) == (N, N, iters+1)

    @test ds["time"][:] == [n*Δt for n in 0:iters]
    @test ds["scalar"][:] == [(n*Δt)^2 for n in 0:iters]

    for n in 0:iters
        @test ds["profile"][:, n+1] == n*Δt .* exp.(znodes(Cell, grid))
        @test ds["slice"][:, :, n+1] == n*Δt .* (   sin.(xnodes(Cell, grid, reshape=true)[:, :, 1])
                                                 .* cos.(ynodes(Face, grid, reshape=true)[:, :, 1]))
    end

    close(simulation.output_writers[:food])

    return nothing
end

function run_jld2_file_splitting_tests(arch)
    model = IncompressibleModel(grid=RegularCartesianGrid(size=(16, 16, 16), extent=(1, 1, 1)))
    simulation = Simulation(model, Δt=1, stop_iteration=10)

    u(model) = Array(model.velocities.u.data.parent)
    fields = Dict(:u => u)

    function fake_bc_init(file, model)
        file["boundary_conditions/fake"] = π
    end

    ow = JLD2OutputWriter(model, fields; dir=".", prefix="test", iteration_interval=1,
                          init=fake_bc_init, including=[:grid],
                          max_filesize=200KiB, force=true)

    push!(simulation.output_writers, ow)

    # 531 KiB of output will be written which should get split into 3 files.
    run!(simulation)

    # Test that files has been split according to size as expected.
    @test filesize("test_part1.jld2") > 200KiB
    @test filesize("test_part2.jld2") > 200KiB
    @test filesize("test_part3.jld2") < 200KiB

    for n in string.(1:3)
        filename = "test_part" * n * ".jld2"
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

"""
Run two coarse rising thermal bubble simulations and make sure that when
restarting from a checkpoint, the restarted simulation matches the non-restarted
simulation to machine precision.
"""
function run_thermal_bubble_checkpointer_tests(arch)
    Nx, Ny, Nz = 16, 16, 16
    Lx, Ly, Lz = 100, 100, 100
    Δt = 6

    grid = RegularCartesianGrid(size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz))
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
    run!(true_simulation)

    checkpointed_simulation = Simulation(checkpointed_model, Δt=Δt, stop_iteration=5)
    checkpointer = Checkpointer(checkpointed_model, iteration_interval=5, force=true)
    push!(checkpointed_simulation.output_writers, checkpointer)

    # Checkpoint should be saved as "checkpoint5.jld" after the 5th iteration.
    run!(checkpointed_simulation)

    # Remove all knowledge of the checkpointed model.
    checkpointed_model = nothing

    # model_kwargs = Dict{Symbol, Any}(:boundary_conditions => SolutionBoundaryConditions(grid))
    restored_model = restore_from_checkpoint("checkpoint_iteration5.jld2")

    for n in 1:4
        time_step!(restored_model, Δt, euler=false)
    end

    rm("checkpoint_iteration0.jld2", force=true)
    rm("checkpoint_iteration5.jld2", force=true)

    # Now the true_model and restored_model should be identical.
    CUDA.@allowscalar begin
        @test all(restored_model.velocities.u.data     .≈ true_model.velocities.u.data)
        @test all(restored_model.velocities.v.data     .≈ true_model.velocities.v.data)
        @test all(restored_model.velocities.w.data     .≈ true_model.velocities.w.data)
        @test all(restored_model.tracers.T.data        .≈ true_model.tracers.T.data)
        @test all(restored_model.tracers.S.data        .≈ true_model.tracers.S.data)
        @test all(restored_model.timestepper.Gⁿ.u.data .≈ true_model.timestepper.Gⁿ.u.data)
        @test all(restored_model.timestepper.Gⁿ.v.data .≈ true_model.timestepper.Gⁿ.v.data)
        @test all(restored_model.timestepper.Gⁿ.w.data .≈ true_model.timestepper.Gⁿ.w.data)
        @test all(restored_model.timestepper.Gⁿ.T.data .≈ true_model.timestepper.Gⁿ.T.data)
        @test all(restored_model.timestepper.Gⁿ.S.data .≈ true_model.timestepper.Gⁿ.S.data)
    end

    return nothing
end

function run_checkpoint_with_function_bcs_tests(arch)
    grid = RegularCartesianGrid(size=(16, 16, 16), extent=(1, 1, 1))

    @inline some_flux(x, y, t) = 2x + exp(y)
    some_flux_bf = BoundaryFunction{:z, Cell, Cell}(some_flux)
    top_u_bc = top_T_bc = FluxBoundaryCondition(some_flux_bf)
    u_bcs = UVelocityBoundaryConditions(grid, top=top_u_bc)
    T_bcs = TracerBoundaryConditions(grid, top=top_T_bc)

    model = IncompressibleModel(architecture=arch, grid=grid, boundary_conditions=(u=u_bcs, T=T_bcs))
    set!(model, u=π/2, v=ℯ, T=Base.MathConstants.γ, S=Base.MathConstants.φ)

    checkpointer = Checkpointer(model)
    write_output(model, checkpointer)
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
    @test u.boundary_conditions.z.right.condition isa BoundaryFunction
    @test u.boundary_conditions.z.right.condition.func(1, 2, 3) == some_flux(1, 2, 3)

    @test T.boundary_conditions.x.left  isa PBC
    @test T.boundary_conditions.x.right isa PBC
    @test T.boundary_conditions.y.left  isa PBC
    @test T.boundary_conditions.y.right isa PBC
    @test T.boundary_conditions.z.left  isa ZFBC
    @test T.boundary_conditions.z.right isa FBC
    @test T.boundary_conditions.z.right.condition isa BoundaryFunction
    @test T.boundary_conditions.z.right.condition.func(1, 2, 3) == some_flux(1, 2, 3)

    # Test that the restored model can be time stepped
    time_step!(properly_restored_model, 1)
    @test properly_restored_model isa IncompressibleModel

    return nothing
end

function run_cross_architecture_checkpointer_tests(arch1, arch2)
    grid = RegularCartesianGrid(size=(16, 16, 16), extent=(1, 1, 1))
    model = IncompressibleModel(architecture=arch1, grid=grid)
    set!(model, u=π/2, v=ℯ, T=Base.MathConstants.γ, S=Base.MathConstants.φ)

    checkpointer = Checkpointer(model)
    write_output(model, checkpointer)
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

function dependencies_added_correctly!(model, windowed_time_average, output_writer)

    model.clock.iteration = 0
    model.clock.time = 0.0
    simulation = Simulation(model, Δt=1.0, stop_iteration=1)
    push!(simulation.output_writers, output_writer)
    run!(simulation)

    return windowed_time_average ∈ values(simulation.diagnostics)
end

function jld2_time_averaging_window(model)

    model.clock.iteration = 0
    model.clock.time = 0.0

    output = FieldOutputs(model.velocities)

    jld2_output_writer = JLD2OutputWriter(model, output, time_interval=4.0, time_averaging_window=2.0,
                                          dir=".", prefix="test", force=true)

    outputs_are_time_averaged = Tuple(typeof(out) <: WindowedTimeAverage for out in jld2_output_writer.outputs)

    return all(outputs_are_time_averaged)
end

@testset "Output writers" begin
    @info "Testing output writers..."

    for arch in archs
         @testset "NetCDF [$(typeof(arch))]" begin
             @info "  Testing NetCDF output writer [$(typeof(arch))]..."
             run_thermal_bubble_netcdf_tests(arch)
             run_thermal_bubble_netcdf_tests_with_halos(arch)
             run_netcdf_function_output_tests(arch)
         end

        @testset "JLD2 [$(typeof(arch))]" begin
            @info "  Testing JLD2 output writer [$(typeof(arch))]..."
            run_jld2_file_splitting_tests(arch)
        end

        @testset "Checkpointer [$(typeof(arch))]" begin
            @info "  Testing Checkpointer [$(typeof(arch))]..."
            run_thermal_bubble_checkpointer_tests(arch)
            run_checkpoint_with_function_bcs_tests(arch)
            @hascuda run_cross_architecture_checkpointer_tests(CPU(), GPU())
            @hascuda run_cross_architecture_checkpointer_tests(GPU(), CPU())
        end

        @testset "Output writer averaging and 'diagnostic dependencies' [$(typeof(arch))]" begin
            @info "  Testing output writer time-averaging and diagnostic-dependencies [$(typeof(arch))]..."

            grid = RegularCartesianGrid(size=(16, 16, 16), extent=(1, 1, 1))
            model = IncompressibleModel(architecture=arch, grid=grid)

            @test jld2_time_averaging_window(model)

            windowed_time_average = WindowedTimeAverage(model.velocities.u, time_window=2.0, time_interval=4.0)

            output = Dict("time_average" => windowed_time_average)
            attributes = Dict("time_average" => Dict("longname" => "A time average",  "units" => "arbitrary"))
            dimensions = Dict("time_average" => ("xF", "yC", "zC"))

            # JLD2 dependencies test
            jld2_output_writer = JLD2OutputWriter(model, output, time_interval=4.0, dir=".", prefix="test", force=true)

            @test dependencies_added_correctly!(model, windowed_time_average, jld2_output_writer)

            # NetCDF dependency test
            netcdf_output_writer =
                NetCDFOutputWriter(model, output, time_interval=4.0, filename="test.nc", with_halos=true,
                                   output_attributes=attributes, dimensions=dimensions)

            @test dependencies_added_correctly!(model, windowed_time_average, netcdf_output_writer)
        end
    end
end
