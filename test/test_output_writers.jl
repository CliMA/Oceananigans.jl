using NCDatasets

"""
Run a coarse thermal bubble simulation and save the output to NetCDF at the
10th time step. Then read back the output and test that it matches the model's
state.
"""
function run_thermal_bubble_netcdf_tests(arch)
    Nx, Ny, Nz = 16, 16, 16
    Lx, Ly, Lz = 100, 100, 100

    grid = RegularCartesianGrid(size=(Nx, Ny, Nz), length=(Lx, Ly, Lz))
    closure = ConstantIsotropicDiffusivity(ν=4e-2, κ=4e-2)
    model = IncompressibleModel(architecture=arch, grid=grid, closure=closure)
    simulation = Simulation(model, Δt=6, stop_iteration=10)

    # Add a cube-shaped warm temperature anomaly that takes up the middle 50%
    # of the domain volume.
    i1, i2 = round(Int, Nx/4), round(Int, 3Nx/4)
    j1, j2 = round(Int, Ny/4), round(Int, 3Ny/4)
    k1, k2 = round(Int, Nz/4), round(Int, 3Nz/4)
    model.tracers.T.data[i1:i2, j1:j2, k1:k2] .+= 0.01

    outputs = Dict(
        "v" => model.velocities.v,
        "u" => model.velocities.u,
        "w" => model.velocities.w,
        "T" => model.tracers.T,
        "S" => model.tracers.S
    )

    nc_writer = NetCDFOutputWriter(model, outputs, filename="dump_test.nc", frequency=10)
    push!(simulation.output_writers, nc_writer)

    xC_slice = 1:10
    xF_slice = 2:11
    yC_slice = 10:15
    yF_slice = 1
    zC_slice = 10
    zF_slice = 9:11

    nc_sliced_writer =
        NetCDFOutputWriter(model, outputs, filename="dump_test_sliced.nc", frequency=10,
                           xC=xC_slice, xF=xF_slice, yC=yC_slice,
                           yF=yF_slice, zC=zC_slice, zF=zF_slice)

    push!(simulation.output_writers, nc_sliced_writer)

    run!(simulation)
    close(nc_writer)
    close(nc_sliced_writer)

    ds3 = Dataset("dump_test.nc")
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

    ds2 = Dataset("dump_test_sliced.nc")
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

function run_netcdf_function_output_tests(arch)
    N = 16
    L = 1
    model = IncompressibleModel(grid=RegularCartesianGrid(size=(N, N, N), length=(L, 2L, 3L)))
    simulation = Simulation(model, Δt=1.25, stop_iteration=3)

    # Define scalar, vector, 2D slice, and 3D field outputs
    f(model) = model.clock.time^2
    g(model) = @. model.clock.time * exp(model.grid.zC)
    h(model) = @. model.clock.time * sin(model.grid.xC) * cos(model.grid.yC')

    outputs = Dict("scalar" => f, "profile" => g, "slice" => h)
    dims = Dict("scalar" => (), "profile" => ("zC",), "slice" => ("xC", "yC"))

    output_attributes = Dict(
        "scalar"  => Dict("longname" => "Some scalar", "units" => "bananas"),
        "profile" => Dict("longname" => "Some vertical profile", "units" => "watermelons"),
        "slice"   => Dict("longname" => "Some slice", "units" => "mushrooms")
    )

    global_attributes = Dict("location" => "Bay of Fundy", "onions" => 7)

    simulation.output_writers[:fruits] =
        NetCDFOutputWriter(
            model, outputs; frequency=1, filename="test_function_outputs.nc", dimensions=dims,
            global_attributes=global_attributes, output_attributes=output_attributes)

    run!(simulation)
    close(simulation.output_writers[:fruits])

    ds = Dataset("test_function_outputs.nc", "r")

    @test ds.attrib["location"] == "Bay of Fundy"
    @test ds.attrib["onions"] == 7

    @test length(ds["time"]) == 4
    @test ds["time"][:] == [1.25i for i in 0:3]

    @test ds["scalar"].attrib["longname"] == "Some scalar"
    @test ds["scalar"].attrib["units"] == "bananas"
    @test ds["scalar"][:] == [(1.25i)^2 for i in 0:3]
    @test dimnames(ds["scalar"]) == ("time",)

    @test ds["profile"].attrib["longname"] == "Some vertical profile"
    @test ds["profile"].attrib["units"] == "watermelons"
    @test ds["profile"][:, end] == @. 3.75 * exp(model.grid.zC)
    @test size(ds["profile"]) == (N, 4)
    @test dimnames(ds["profile"]) == ("zC", "time")

    @test ds["slice"].attrib["longname"] == "Some slice"
    @test ds["slice"].attrib["units"] == "mushrooms"
    @test ds["slice"][:, :, end] == @. 3.75 * sin(model.grid.xC) * cos(model.grid.yC')
    @test size(ds["slice"]) == (N, N, 4)
    @test dimnames(ds["slice"]) == ("xC", "yC", "time")

    close(ds)
    return nothing
end

function run_jld2_file_splitting_tests(arch)
    model = IncompressibleModel(grid=RegularCartesianGrid(size=(16, 16, 16), length=(1, 1, 1)))
    simulation = Simulation(model, Δt=1, stop_iteration=10)

    u(model) = Array(model.velocities.u.data.parent)
    fields = Dict(:u => u)

    function fake_bc_init(file, model)
        file["boundary_conditions/fake"] = π
    end

    ow = JLD2OutputWriter(model, fields; dir=".", prefix="test", frequency=1,
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
simulation numerically.
"""
function run_thermal_bubble_checkpointer_tests(arch)
    Nx, Ny, Nz = 16, 16, 16
    Lx, Ly, Lz = 100, 100, 100
    Δt = 6

    grid = RegularCartesianGrid(size=(Nx, Ny, Nz), length=(Lx, Ly, Lz))
    closure = ConstantIsotropicDiffusivity(ν=4e-2, κ=4e-2)
    true_model = IncompressibleModel(architecture=arch, grid=grid, closure=closure)

    # Add a cube-shaped warm temperature anomaly that takes up the middle 50%
    # of the domain volume.
    i1, i2 = round(Int, Nx/4), round(Int, 3Nx/4)
    j1, j2 = round(Int, Ny/4), round(Int, 3Ny/4)
    k1, k2 = round(Int, Nz/4), round(Int, 3Nz/4)
    true_model.tracers.T.data[i1:i2, j1:j2, k1:k2] .+= 0.01

    checkpointed_model = deepcopy(true_model)

    true_simulation = Simulation(true_model, Δt=Δt, stop_iteration=9)
    run!(true_simulation)

    checkpointed_simulation = Simulation(checkpointed_model, Δt=Δt, stop_iteration=5)
    checkpointer = Checkpointer(checkpointed_model, frequency=5, force=true)
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

@testset "Output writers" begin
    @info "Testing output writers..."

    for arch in archs
         @testset "NetCDF [$(typeof(arch))]" begin
             @info "  Testing NetCDF output writer [$(typeof(arch))]..."
             run_thermal_bubble_netcdf_tests(arch)
             run_netcdf_function_output_tests(arch)
         end

        @testset "JLD2 [$(typeof(arch))]" begin
            @info "  Testing JLD2 output writer [$(typeof(arch))]..."
            run_jld2_file_splitting_tests(arch)
        end

        @testset "Checkpointer [$(typeof(arch))]" begin
            @info "  Testing Checkpointer [$(typeof(arch))]..."
            run_thermal_bubble_checkpointer_tests(arch)
        end
    end
end
