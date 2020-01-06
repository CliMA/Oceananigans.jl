using Oceananigans.Grids: unpack_grid

#####
##### CPU channel pressure solver
#####

function ChannelPressureSolver(::CPU, grid, pressure_bcs, planner_flag=FFTW.PATIENT)
    kx², ky², kz² = generate_discrete_eigenvalues(grid, pressure_bcs)
    wavenumbers = (kx² = kx², ky² = ky², kz² = kz²)

    # Storage for RHS and Fourier coefficients is hard-coded to be Float64
    # because of precision issues with Float32.
    # See https://github.com/climate-machine/Oceananigans.jl/issues/55
    storage = zeros(Complex{Float64}, grid.Nx, grid.Ny, grid.Nz)

    @info "Planning transforms for PressureSolver{Channel, CPU}..."
    x_bc, y_bc, z_bc = pressure_bcs.x.left, pressure_bcs.y.left, pressure_bcs.z.left
    FFTx!   = plan_forward_transform(storage, x_bc, 1, planner_flag)
    DCTyz!  = plan_forward_transform(storage, z_bc, [2, 3], planner_flag)
    IFFTx!  = plan_backward_transform(storage, x_bc, 1, planner_flag)
    IDCTyz! = plan_backward_transform(storage, z_bc, [2, 3], planner_flag)
    @info "Planning transforms for PressureSolver{Channel, CPU} done!"

    transforms = ( FFTx! =  FFTx!,  DCTyz! =  DCTyz!,
                  IFFTx! = IFFTx!, IDCTyz! = IDCTyz!)

    return PressureSolver(Channel(), CPU(), wavenumbers, storage, transforms, nothing)
end

function solve_poisson_equation!(solver::PressureSolver{Channel, CPU}, grid)
    Nx, Ny, Nz, _ = unpack_grid(grid)
    kx², ky², kz² = solver.wavenumbers.kx², solver.wavenumbers.ky², solver.wavenumbers.kz²

    # We can use the same storage for the RHS and the solution ϕ.
    RHS, ϕ = solver.storage, solver.storage

    solver.transforms.DCTyz! * RHS  # Calculate DCTʸᶻ(f) in place.
    solver.transforms.FFTx!  * RHS  # Calculate FFTˣ(f) in place.

    # Solve the discrete Poisson equation in spectral space. We are computing
    # the Fourier coefficients of the solution from the Fourier coefficients
    # of the RHS.
    @. ϕ = -RHS / (kx² + ky² + kz²)

    # Setting DC component of the solution (the mean) to be zero. This is also
    # necessary because the source term to the Poisson equation has zero mean
    # and so the DC component comes out to be ∞.
    ϕ[1, 1, 1] = 0

    solver.transforms.IFFTx!  * ϕ  # Calculate IFFTˣ(ϕ) in place.
    solver.transforms.IDCTyz! * ϕ  # Calculate IDCTʸᶻ(ϕ) in place.

    # Must normalize by 2N for each dimension transformed via FFTW.REDFT.
    @. ϕ = ϕ / (4Ny*Nz)

    return nothing
end

#####
##### GPU channel pressure solver
#####

function ChannelPressureSolver(::GPU, grid, pressure_bcs, no_args...)
    Nx, Ny, Nz, _ = unpack_grid(grid)

    kx², ky², kz² = generate_discrete_eigenvalues(grid, pressure_bcs) .|> CuArray
    wavenumbers = (kx² = kx², ky² = ky², kz² = kz²)

    ky⁺ = reshape(0:Ny-1,       1, Ny, 1)
    kz⁺ = reshape(0:Nz-1,       1, 1, Nz)
    ky⁻ = reshape(0:-1:-(Ny-1), 1, Ny, 1)
    kz⁻ = reshape(0:-1:-(Nz-1), 1, 1, Nz)

    ω_4Ny⁺ = ω.(4Ny, ky⁺) |> CuArray
    ω_4Nz⁺ = ω.(4Nz, kz⁺) |> CuArray
    ω_4Ny⁻ = ω.(4Ny, ky⁻) |> CuArray
    ω_4Nz⁻ = ω.(4Nz, kz⁻) |> CuArray

    # Indices used when we need views with reverse indexing but index N+1 should
    # return a 0. This can't be enforced using views so we just map N+1 to 1,
    # and use masks M_ky and M_kz to enforce that the value at N+1 is 0.
    r_y_inds = [1, collect(Ny:-1:2)...] |> CuArray
    r_z_inds = [1, collect(Nz:-1:2)...] |> CuArray

    # Masks that are useful for writing broadcast operations involving arrays
    # with reversed indices.
    M_ky = ones(1, Ny, 1) |> CuArray
    M_kz = ones(1, 1, Nz) |> CuArray

    M_ky[1] = 0
    M_kz[1] = 0

    constants = (ω_4Ny⁺ = ω_4Ny⁺, ω_4Nz⁺ = ω_4Nz⁺, ω_4Ny⁻ = ω_4Ny⁻, ω_4Nz⁻ = ω_4Nz⁻,
                 r_y_inds = r_y_inds, r_z_inds = r_z_inds,
                 M_ky = M_ky, M_kz = M_kz)

    # Storage for RHS and Fourier coefficients is hard-coded to be Float64
    # because of precision issues with Float32.
    # See https://github.com/climate-machine/Oceananigans.jl/issues/55
    storage1 = zeros(Complex{Float64}, Nx, Ny, Nz) |> CuArray

    # For solving the Poisson equation with PNN boundary conditions, we need a
    # second storage/buffer array to perform the 2D fast cosine transform. Maybe
    # we can get around this but for now we'll just use up a second array.
    storage2 = zeros(Complex{Float64}, Nx, Ny, Nz) |> CuArray

    storage = (storage1 = storage1, storage2 = storage2)

    @info "Planning transforms for PressureSolver{Channel, GPU}..."
    x_bc, y_bc, z_bc = pressure_bcs.x.left, pressure_bcs.y.left, pressure_bcs.z.left
    FFTx!   = plan_forward_transform(storage1, x_bc, 1)
    FFTyz!  = plan_forward_transform(storage1, z_bc, [2, 3])
    IFFTx!  = plan_backward_transform(storage1, x_bc, 1)
    IFFTyz! = plan_backward_transform(storage1, z_bc, [2, 3])
    @info "Planning transforms for PressureSolver{Channel, GPU} done!"

    transforms = ( FFTx! =  FFTx!,  FFTyz! =  FFTyz!,
                  IFFTx! = IFFTx!, IFFTyz! = IFFTyz!)

    return PressureSolver(Channel(), GPU(), wavenumbers, storage, transforms, constants)
end

function solve_poisson_equation!(solver::PressureSolver{Channel, GPU}, grid)
    Nx, Ny, Nz, _ = unpack_grid(grid)

    kx², ky², kz² = solver.kx², solver.ky², solver.kz²
    ω_4Ny⁺, ω_4Ny⁻ = solver.constants.ω_4Ny⁺, solver.constants.ω_4Ny⁻
    ω_4Nz⁺, ω_4Nz⁻ = solver.constants.ω_4Nz⁺, solver.constants.ω_4Nz⁻
    r_y_inds, r_z_inds = solver.constants.r_y_inds, solver.constants.r_z_inds
    M_ky, M_kz = solver.constants.M_ky, solver.constants.M_kz

    # We can use the same storage for the RHS and the solution ϕ.
    RHS =  ϕ = solver.storage.storage1

    B = solver.storage.storage2  # Complex buffer storage.

    # Calculate DCTʸᶻ(f) in place using the FFT.
    solver.transforms.FFTyz! * RHS

    RHS⁻ = view(RHS, 1:Nx, r_y_inds, 1:Nz)
    @. B = 2 * real(ω_4Nz⁺ * (ω_4Ny⁺ * RHS + ω_4Ny⁻ * RHS⁻))

    solver.transforms.FFTx! * B # Calculate FFTˣ(f) in place.

    @. B = -B / (kx² + ky² + kz²)

    B[1, 1, 1] = 0  # Setting DC component of the solution (the mean) to be zero.

    solver.IFFTx! * B  # Calculate IFFTˣ(ϕ̂) in place.

    # Calculate IDCTʸᶻ(ϕ̂) in place using the FFT.
    B⁻⁺ = view(B, 1:Nx, r_y_inds, 1:Nz)
    B⁺⁻ = view(B, 1:Nx, 1:Ny, r_z_inds)
    B⁻⁻ = view(B, 1:Nx, r_y_inds, r_z_inds)

    @. ϕ = 1/4 *  ω_4Ny⁻ * ω_4Nz⁻ * ((B - M_ky * M_kz * B⁻⁻) - im*(M_kz * B⁺⁻ + M_ky * B⁻⁺))

    solver.transforms.IFFTyz! * ϕ

    return nothing
end
