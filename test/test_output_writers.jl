"""
Run two coarse rising thermal bubble simulations and make sure that when
restarting from a checkpoint, the restarted simulation matches the non-restarted
simulation numerically.
"""
function run_thermal_bubble_checkpointer_tests()
    Nx, Ny, Nz = 16, 16, 16
    Lx, Ly, Lz = 100, 100, 100
    Δt = 6

    true_model = Model(N=(Nx, Ny, Nz), L=(Lx, Ly, Lz), ν=4e-2, κ=4e-2)

    # Add a cube-shaped warm temperature anomaly that takes up the middle 50%
    # of the domain volume.
    i1, i2 = round(Int, Nx/4), round(Int, 3Nx/4)
    j1, j2 = round(Int, Ny/4), round(Int, 3Ny/4)
    k1, k2 = round(Int, Nz/4), round(Int, 3Nz/4)
    true_model.tracers.T.data[i1:i2, j1:j2, k1:k2] .+= 0.01

    checkpointed_model = deepcopy(true_model)

    time_step!(true_model, 10, Δt)

    checkpointer = Checkpointer(dir=".", prefix="test_", frequency=5, padding=1)
    push!(checkpointed_model.output_writers, checkpointer)

    # Checkpoint should be saved as "test_model_checkpoint_5.jld" after the
    # 5th iteration.
    time_step!(checkpointed_model, 5, Δt)

    # Remove all knowledge of the checkpointed model.
    checkpointed_model = nothing

    restored_model = restore_from_checkpoint("test_model_checkpoint_5.jld")

    time_step!(restored_model, 5, Δt)

    # Now the true_model and restored_model should be identical.
    @test all(restored_model.velocities.u.data .≈ true_model.velocities.u.data)
    @test all(restored_model.velocities.v.data .≈ true_model.velocities.v.data)
    @test all(restored_model.velocities.w.data .≈ true_model.velocities.w.data)
    @test all(restored_model.tracers.T.data .≈ true_model.tracers.T.data)
    @test all(restored_model.tracers.S.data .≈ true_model.tracers.S.data)
    @test all(restored_model.G.Gu.data .≈ true_model.G.Gu.data)
    @test all(restored_model.G.Gv.data .≈ true_model.G.Gv.data)
    @test all(restored_model.G.Gw.data .≈ true_model.G.Gw.data)
    @test all(restored_model.G.GT.data .≈ true_model.G.GT.data)
    @test all(restored_model.G.GS.data .≈ true_model.G.GS.data)
end

"""
Run a coarse thermal bubble simulation and save the output to NetCDF at the
10th time step. Then read back the output and test that it matches the model's
state.
"""
function run_thermal_bubble_netcdf_tests()
    Nx, Ny, Nz = 16, 16, 16
    Lx, Ly, Lz = 100, 100, 100
    Δt = 6

    model = Model(N=(Nx, Ny, Nz), L=(Lx, Ly, Lz), ν=4e-2, κ=4e-2)

    # Add a cube-shaped warm temperature anomaly that takes up the middle 50%
    # of the domain volume.
    i1, i2 = round(Int, Nx/4), round(Int, 3Nx/4)
    j1, j2 = round(Int, Ny/4), round(Int, 3Ny/4)
    k1, k2 = round(Int, Nz/4), round(Int, 3Nz/4)
    model.tracers.T.data[i1:i2, j1:j2, k1:k2] .+= 0.01

    nc_writer = NetCDFOutputWriter(dir=".", prefix="test_", frequency=10, padding=1)
    push!(model.output_writers, nc_writer)

    time_step!(model, 10, Δt)

    u = read_output(nc_writer, "u", 10)
    v = read_output(nc_writer, "v", 10)
    w = read_output(nc_writer, "w", 10)
    T = read_output(nc_writer, "T", 10)
    S = read_output(nc_writer, "S", 10)

    @test all(u .≈ model.velocities.u.data)
    @test all(v .≈ model.velocities.v.data)
    @test all(w .≈ model.velocities.w.data)
    @test all(T .≈ model.tracers.T.data)
    @test all(S .≈ model.tracers.S.data)
end
