#####
##### CPU horizontally periodic pressure solver for regular grids
#####

function HorizontallyPeriodicPressureSolver(::CPU, grid, pressure_bcs, planner_flag=FFTW.PATIENT)
    kx², ky², kz² = generate_discrete_eigenvalues(grid, pressure_bcs)
    wavenumbers = (kx² = kx², ky² = ky², kz² = kz²)

    # Storage for RHS and Fourier coefficients is hard-coded to be Float64
    # because of precision issues with Float32.
    # See https://github.com/climate-machine/Oceananigans.jl/issues/55
    storage = zeros(Complex{Float64}, grid.Nx, grid.Ny, grid.Nz)

    @debug "Planning transforms for PressureSolver{HorizontallyPeriodic, CPU}..."
    x_bc, y_bc, z_bc = pressure_bcs.x.left, pressure_bcs.y.left, pressure_bcs.z.left
    FFTxy!  = plan_forward_transform(storage, x_bc, [1, 2], planner_flag)
    DCTz!   = plan_forward_transform(storage, z_bc, 3, planner_flag)
    IFFTxy! = plan_backward_transform(storage, x_bc, [1, 2], planner_flag)
    IDCTz!  = plan_backward_transform(storage, z_bc, 3, planner_flag)
    @debug "Planning transforms for PressureSolver{HorizontallyPeriodic, CPU} done!"

    transforms = ( FFTxy! =  FFTxy!,  DCTz! =  DCTz!,
                  IFFTxy! = IFFTxy!, IDCTz! = IDCTz!)

    return PressureSolver(HorizontallyPeriodic(), CPU(), wavenumbers, storage, transforms, nothing)
end

"""
    solve_poisson_equation!(solver::PressureSolver{HorizontallyPeriodic, CPU}, grid)

Solve Poisson equation on a uniform staggered grid (Arakawa C-grid) with
appropriate boundary conditions as specified by `solver.bcs` using planned FFTs
and DCTs. The right-hand-side RHS is stored in solver.storage which the solver
mutates to produce the solution, so it will also be stored in solver.storage.

We should describe the algorithm in detail in the documentation.
"""
function solve_poisson_equation!(solver::PressureSolver{HorizontallyPeriodic, CPU}, grid)
    Nx, Ny, Nz, _ = unpack_grid(grid)
    kx², ky², kz² = solver.wavenumbers.kx², solver.wavenumbers.ky², solver.wavenumbers.kz²

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

    solver.transforms.IFFTxy! * ϕ  # Calculate IFFTˣʸ(ϕ) in place.
    solver.transforms.IDCTz!  * ϕ  # Calculate IDCTᶻ(ϕ) in place.

    # Must normalize by 2N for each dimension transformed via FFTW.REDFT.
    @. ϕ = ϕ / (2Nz)

    return nothing
end

#####
##### GPU horizontally periodic pressure solver for regular grids
#####

function HorizontallyPeriodicPressureSolver(::GPU, grid, pressure_bcs, no_args...)
    Nx, Ny, Nz, _ = unpack_grid(grid)

    kx², ky², kz² = generate_discrete_eigenvalues(grid, pressure_bcs) .|> CuArray
    wavenumbers = (kx² = kx², ky² = ky², kz² = kz²)

    kz⁺ = reshape(0:Nz-1,       1, 1, Nz)
    kz⁻ = reshape(0:-1:-(Nz-1), 1, 1, Nz)

    ω_4Nz⁺ = ω.(4Nz, kz⁺) |> CuArray
    ω_4Nz⁻ = ω.(4Nz, kz⁻) |> CuArray

    # The zeroth coefficient of the IDCT (DCT-III or FFTW.REDFT01) is not
    # multiplied by 2. For some reason, we only need to account for this when
    # doing a 1D IDCT (for PPN boundary conditions) but not for a 2D IDCT. It's
    # possible that the masks are effectively doing this job.
    ω_4Nz⁻[1] *= 1/2

    constants = (ω_4Nz⁺ = ω_4Nz⁺, ω_4Nz⁻ = ω_4Nz⁻)

    # Storage for RHS and Fourier coefficients is hard-coded to be Float64
    # because of precision issues with Float32.
    # See https://github.com/climate-machine/Oceananigans.jl/issues/55
    storage = zeros(Complex{Float64}, Nx, Ny, Nz) |> CuArray

    @debug "Planning transforms for PressureSolver{HorizontallyPeriodic, GPU}..."
    x_bc, y_bc, z_bc = pressure_bcs.x.left, pressure_bcs.y.left, pressure_bcs.z.left
    FFTxy!  = plan_forward_transform(storage, x_bc, [1, 2])
    FFTz!   = plan_forward_transform(storage, z_bc, 3)
    IFFTxy! = plan_backward_transform(storage, x_bc, [1, 2])
    IFFTz!  = plan_backward_transform(storage, z_bc, 3)
    @debug "Planning transforms for PressureSolver{HorizontallyPeriodic, GPU} done!"

    transforms = ( FFTxy! =  FFTxy!,  FFTz! =  FFTz!,
                  IFFTxy! = IFFTxy!, IFFTz! = IFFTz!)

    return PressureSolver(HorizontallyPeriodic(), GPU(), wavenumbers, storage, transforms, constants)
end

"""
    solve_poisson_equation!(solver::PressureSolver{HorizontallyPeriodic, GPU}, grid)

Similar to solve_poisson_equation!(solver::HPPS{CPU}, grid) except that since
the discrete cosine transform is not available through cuFFT, we perform our
own fast cosine transform (FCT) via an algorithm that utilizes the FFT.

Note that for the FCT algorithm to work, the input must have been permuted along
the dimension the FCT is to be calculated by ordering the odd elements first
followed by the even elements. For example,

    [a, b, c, d, e, f, g, h] -> [a, c, e, g, h, f, d, b]

The output will be permuted in this way and so the permutation must be undone.

We should describe the algorithm in detail in the documentation.
"""
function solve_poisson_equation!(solver::PressureSolver{HorizontallyPeriodic, GPU}, grid)
    kx², ky², kz² = solver.wavenumbers.kx², solver.wavenumbers.ky², solver.wavenumbers.kz²
    ω_4Nz⁺, ω_4Nz⁻ = solver.constants.ω_4Nz⁺, solver.constants.ω_4Nz⁻

    # We can use the same storage for the RHS and the solution ϕ.
    RHS, ϕ = solver.storage, solver.storage

    # Calculate DCTᶻ(f) in place using the FFT.
    solver.transforms.FFTz! * RHS
    @. RHS = 2 * real(ω_4Nz⁺ * RHS)

    solver.transforms.FFTxy! * RHS  # Calculate FFTˣʸ(f) in place.

    # Solve the discrete Poisson equation in spectral space. We are essentially
    # computing the Fourier coefficients of the solution from the Fourier
    # coefficients of the RHS.
    @. ϕ = -RHS / (kx² + ky² + kz²)

    # Setting DC component of the solution (the mean) to be zero. This is also
    # necessary because the source term to the Poisson equation has zero mean
    # and so the DC component comes out to be ∞.
    ϕ[1, 1, 1] = 0

    solver.transforms.IFFTxy! * ϕ  # Calculate IFFTˣʸ(ϕ̂) in place.

    # Calculate IDCTᶻ(ϕ̂) in place using the FFT.
    ϕ .*= ω_4Nz⁻
    solver.transforms.IFFTz! * ϕ

    return nothing
end

#####
##### Horizontally periodic pressure solver for vertically stretched grids
#####

# function HorizontallyPeriodicPressureSolver(arch, grid::VerticallyStretchedCartesianGrid, pressure_bcs, planner_flag=nothing)
#     kx², ky², _ = generate_discrete_eigenvalues(grid, pressure_bcs)
#     wavenumbers = (kx² = kx², ky² = ky²)
#
#     if isnothing(planner_flag) && arch isa CPU
#         planner_flag = FFTW.PATIENT
#     end
#
#     storage = zeros(Complex{Float64}, grid.Nx, grid.Ny, grid.Nz)
#
#     @debug "Planning transforms for PressureSolver{HorizontallyPeriodic, $(typeof(arch)), VerticallyStretchedCartesianGrid}..."
#     x_bc, y_bc = pressure_bcs.x.left, pressure_bcs.y.left
#     FFTxy!  = plan_forward_transform(storage, x_bc, [1, 2], planner_flag)
#     IFFTxy! = plan_backward_transform(storage, x_bc, [1, 2], planner_flag)
#     @debug "Planning transforms for PressureSolver{HorizontallyPeriodic, $(typeof(arch)), VerticallyStretchedCartesianGrid} done!"
#
#     # Set up batched tridiagonal solver.
#     dl = [1/grid.ΔzF[k] for k in 1:Nz-1]
#     du = copy(ld)
#
#     # Diagonal (different for each i,j)
#     @inline δ(k, ΔzF, ΔzC, kx², ky²) = - (1/ΔzF[k-1] + 1/ΔzF[k]) - ΔzC[k] * (kx² + ky²)
#
#     d = zeros(Nx, Ny, Nz)
#     for i in 1:Nx, j in 1:Ny
#         d[i, j, 1] = -1/ΔzF[1] - ΔzC[1] * (kx²[i] + ky²[j])
#         d[i, j, 2:Nz-1] .= [δ(k, ΔzF, ΔzC, kx²[i], ky²[j]) for k in 2:Nz-1]
#         d[i, j, Nz] = -1/ΔzF[Nz-1] - ΔzC[Nz] * (kx²[i] + ky²[j])
#     end
#
#     bt_solver = BatchedTridiagonalSolver(arch, dl=dl, d=d, du=du, f=storage, grid=grid)
#
#     transforms = (FFTxy! = FFTxy!, IFFTxy! = IFFTxy!, batched_tridiagonal_solver = bt_solver)
#
#     return PressureSolver(HorizontallyPeriodic(), CPU(), wavenumbers, storage, transforms, nothing)
# end
