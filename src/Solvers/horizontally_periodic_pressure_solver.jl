using Oceananigans.Grids: unpack_grid

struct HorizontallyPeriodicPressureSolver{A, R, T, C} <: AbstractPressureSolver{A}
          arch :: A
           kx² :: R
           ky² :: R
           kz² :: R
       storage :: C
    transforms :: T
end

const HPPS = HorizontallyPeriodicPressureSolver

function HorizontallyPeriodicPressureSolver(::CPU, grid, pressure_bcs, planner_flag=FFTW.PATIENT)
    x_bc, y_bc, z_bc = pressure_bcs.x.left, pressure_bcs.y.left, pressure_bcs.z.left

    kx², ky², kz² = generate_discrete_eigenvalues(grid, pressure_bcs)

    # Storage for RHS and Fourier coefficients is hard-coded to be Float64
    # because of precision issues with Float32.
    # See https://github.com/climate-machine/Oceananigans.jl/issues/55
    storage = zeros(Complex{Float64}, grid.Nx, grid.Ny, grid.Nz)

    @info "Planning transforms for HorizontallyPeriodicPressureSolver..."
    FFTxy!  = plan_forward_transform(storage, x_bc, [1, 2], planner_flag)
    DCTz!   = plan_forward_transform(storage, z_bc, 3, planner_flag)
    IFFTxy! = plan_backward_transform(storage, x_bc, [1, 2], planner_flag)
    IDCTz!  = plan_backward_transform(storage, z_bc, 3, planner_flag)
    @info "Planning transforms for HorizontallyPeriodicPressureSolver done!"

    transforms = (FFTxy! = FFTxy!, DCTz! = DCTz!,
                  IFFTxy! = IFFTxy!, IDCTz! = IDCTz!)

    return HorizontallyPeriodicPressureSolver(CPU(), kx², ky², kz², storage, transforms)
end

function solve_poisson_equation!(output, solver::HPPS, grid)
    Nx, Ny, Nz, _ = unpack_grid(grid)
    kx², ky², kz² = solver.kx², solver.ky², solver.kz²

    # We can use the same storage for the RHS and the solution ϕ.
    RHS, ϕ = solver.storage, solver.storage

    solver.transforms.DCTz!  * RHS  # Calculate DCTᶻ(f) in place.
    solver.transforms.FFTxy! * RHS  # Calculate FFTˣʸ(f) in place.

    # Solve the discrete Poisson equation in spectral space. We are computing
    # the Fourier coefficients of the solution from the Fourier coefficients
    # of the RHS.
    @. ϕ = -RHS / (kx² + ky² + kz²)

    # Setting DC component of the solution (the mean) to be zero. This is also
    # necessary because the source term to the Poisson equation has zero mean
    # and so the DC component comes out to be ∞.
    ϕ[1, 1, 1] = 0

    solver.transforms.IFFT! * ϕ  # Calculate IFFTˣʸ(ϕ) in place.
    solver.transforms.IDCT! * ϕ  # Calculate IDCTᶻ(ϕ) in place.

    # Must normalize by 2N for each dimension transformed via FFTW.REDFT.
    @. ϕ = ϕ / (2Nz)

    return nothing
end
