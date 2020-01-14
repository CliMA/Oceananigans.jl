#####
##### CPU box pressure solver
#####

function BoxPressureSolver(::CPU, grid, pressure_bcs, planner_flag=FFTW.PATIENT)
    kx², ky², kz² = generate_discrete_eigenvalues(grid, pressure_bcs)
    wavenumbers = (kx² = kx², ky² = ky², kz² = kz²)

    # Storage for RHS and Fourier coefficients is hard-coded to be Float64
    # because of precision issues with Float32.
    # See https://github.com/climate-machine/Oceananigans.jl/issues/55
    storage = zeros(Complex{Float64}, grid.Nx, grid.Ny, grid.Nz)

    @debug "Planning transforms for PressureSolver{Box, CPU}..."
    x_bc, y_bc, z_bc = pressure_bcs.x.left, pressure_bcs.y.left, pressure_bcs.z.left
    DCTxyz!  =  plan_forward_transform(storage, z_bc, [1, 2, 3], planner_flag)
    IDCTxyz! = plan_backward_transform(storage, z_bc, [1, 2, 3], planner_flag)
    @debug "Planning transforms for PressureSolver{Box, CPU} done!"

    transforms = (DCTxyz! =  DCTxyz!, IDCTxyz! = IDCTxyz!)

    return PressureSolver(Box(), CPU(), wavenumbers, storage, transforms, nothing)
end

function solve_poisson_equation!(solver::PressureSolver{Box, CPU}, grid)
    Nx, Ny, Nz, _ = unpack_grid(grid)
    kx², ky², kz² = solver.wavenumbers.kx², solver.wavenumbers.ky², solver.wavenumbers.kz²

    # We can use the same storage for the RHS and the solution ϕ.
    RHS, ϕ = solver.storage, solver.storage

    solver.transforms.DCTxyz! * RHS  # Calculate DCTˣʸᶻ(f) in place.

    # Solve the discrete Poisson equation in spectral space. We are computing
    # the Fourier coefficients of the solution from the Fourier coefficients
    # of the RHS.
    @. ϕ = -RHS / (kx² + ky² + kz²)

    # Setting DC component of the solution (the mean) to be zero. This is also
    # necessary because the source term to the Poisson equation has zero mean
    # and so the DC component comes out to be ∞.
    ϕ[1, 1, 1] = 0

    solver.transforms.IDCTxyz! * ϕ  # Calculate IDCTˣʸᶻ(ϕ) in place.

    # Must normalize by 2N for each dimension transformed via FFTW.REDFT.
    @. ϕ = ϕ / (8Nx*Ny*Nz)

    return nothing
end

#####
##### GPU channel pressure solver
#####

function BoxPressureSolver(::GPU, grid, pressure_bcs, no_args...)
    throw(ArgumentError("Box pressure solver not implemented for GPUs yet :("))
end
