"""
Run two coarse rising thermal bubble simulations and make sure that when
restarting from a checkpoint, the restarted simulation matches the non-restarted
simulation numerically.
"""
function run_basic_checkpointer_tests()
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

    time_step!(true_model, 5, Δt)

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
end
