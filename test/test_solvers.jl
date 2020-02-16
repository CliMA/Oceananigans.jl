using LinearAlgebra
using Oceananigans.Architectures: array_type
using Oceananigans.TimeSteppers: _compute_w_from_continuity!

function can_solve_single_tridiagonal_system(arch, N)
    ArrayType = array_type(arch)

    a = rand(N-1)
    b = 3 .+ rand(N) # +3 to ensure diagonal dominance.
    c = rand(N-1)
    f = rand(N)

    # Solve the system with backslash on the CPU to avoid scalar operations on the GPU.
    M = Tridiagonal(a, b, c)
    ϕ_correct = M \ f

    # Convert to CuArray if needed.
    a, b, c, f = ArrayType.([a, b, c, f])

    ϕ = reshape(zeros(N), (1, 1, N)) |> ArrayType

    grid = RegularCartesianGrid(size=(1, 1, N), length=(1, 1, 1))
    btsolver = BatchedTridiagonalSolver(arch; dl=a, d=b, du=c, f=f, grid=grid)

    solve_batched_tridiagonal_system!(ϕ, arch, btsolver)

    return Array(ϕ[:]) ≈ ϕ_correct
end

function can_solve_single_tridiagonal_system_with_functions(arch, N)
    ArrayType = array_type(arch)

    grid = RegularCartesianGrid(size=(1, 1, N), length=(1, 1, 1))

    a = rand(N-1)
    c = rand(N-1)

    @inline b(i, j, k, grid, p) = 3 .+ cos(2π*grid.zC[k])  # +3 to ensure diagonal dominance.
    @inline f(i, j, k, grid, p) = sin(2π*grid.zC[k])

    bₐ = [b(1, 1, k, grid, nothing) for k in 1:N]
    fₐ = [f(1, 1, k, grid, nothing) for k in 1:N]

    # Solve the system with backslash on the CPU to avoid scalar operations on the GPU.
    M = Tridiagonal(a, bₐ, c)
    ϕ_correct = M \ fₐ

    # Convert to CuArray if needed.
    a, c = ArrayType.([a, c])

    ϕ = reshape(zeros(N), (1, 1, N)) |> ArrayType

    btsolver = BatchedTridiagonalSolver(arch; dl=a, d=b, du=c, f=f, grid=grid)

    solve_batched_tridiagonal_system!(ϕ, arch, btsolver)

    return Array(ϕ[:]) ≈ ϕ_correct
end

function can_solve_batched_tridiagonal_system_with_3D_RHS(arch, Nx, Ny, Nz)
    ArrayType = array_type(arch)

    a = rand(Nz-1)
    b = 3 .+ rand(Nz) # +3 to ensure diagonal dominance.
    c = rand(Nz-1)
    f = rand(Nx, Ny, Nz)

    M = Tridiagonal(a, b, c)
    ϕ_correct = zeros(Nx, Ny, Nz)

    # Solve the systems with backslash on the CPU to avoid scalar operations on the GPU.
    for i = 1:Nx, j = 1:Ny
        ϕ_correct[i, j, :] .= M \ f[i, j, :]
    end

    # Convert to CuArray if needed.
    a, b, c, f = ArrayType.([a, b, c, f])

    grid = RegularCartesianGrid(size=(Nx, Ny, Nz), length=(1, 1, 1))
    btsolver = BatchedTridiagonalSolver(arch; dl=a, d=b, du=c, f=f, grid=grid)

    ϕ = zeros(Nx, Ny, Nz) |> ArrayType

    solve_batched_tridiagonal_system!(ϕ, arch, btsolver)

    return Array(ϕ) ≈ ϕ_correct
end

function can_solve_batched_tridiagonal_system_with_3D_functions(arch, Nx, Ny, Nz)
    ArrayType = array_type(arch)

    grid = RegularCartesianGrid(size=(Nx, Ny, Nz), length=(1, 1, 1))

    a = rand(Nz-1)
    c = rand(Nz-1)

    @inline b(i, j, k, grid, p) = 3 + grid.xC[i]*grid.yC[j] * cos(2π*grid.zC[k])
    @inline f(i, j, k, grid, p) = (grid.xC[i] + grid.yC[j]) * sin(2π*grid.zC[k])

    ϕ_correct = zeros(Nx, Ny, Nz)

    # Solve the system with backslash on the CPU to avoid scalar operations on the GPU.
    for i = 1:Nx, j = 1:Ny
        bₐ = [b(i, j, k, grid, nothing) for k in 1:Nz]
        M = Tridiagonal(a, bₐ, c)

        fₐ = [f(i, j, k, grid, nothing) for k in 1:Nz]
        ϕ_correct[i, j, :] .= M \ fₐ
    end

    # Convert to CuArray if needed.
    a, c = ArrayType.([a, c])

    btsolver = BatchedTridiagonalSolver(arch; dl=a, d=b, du=c, f=f, grid=grid)

    ϕ = zeros(Nx, Ny, Nz) |> ArrayType
    solve_batched_tridiagonal_system!(ϕ, arch, btsolver)

    return Array(ϕ) ≈ ϕ_correct
end

function vertically_stretched_poisson_solver_correct_answer(arch, Nx, Ny, zF)
    Lx, Ly, Lz = 1, 1, zF[end]
    Δx, Δy = Lx/Nx, Ly/Ny

    #####
    ##### Vertically stretched operators
    #####

    @inline δx_caa(i, j, k, f) = @inbounds f[i+1, j, k] - f[i, j, k]
    @inline δy_aca(i, j, k, f) = @inbounds f[i, j+1, k] - f[i, j, k]
    @inline δz_aac(i, j, k, f) = @inbounds f[i, j, k+1] - f[i, j, k]

    @inline ∂x_caa(i, j, k, Δx,  f) = δx_caa(i, j, k, f) / Δx
    @inline ∂y_aca(i, j, k, Δy,  f) = δy_aca(i, j, k, f) / Δy
    @inline ∂z_aac(i, j, k, ΔzF, f) = δz_aac(i, j, k, f) / ΔzF[k]

    @inline ∂x²(i, j, k, Δx, f)       = (∂x_caa(i, j, k, Δx, f)  - ∂x_caa(i-1, j, k, Δx, f))  / Δx
    @inline ∂y²(i, j, k, Δy, f)       = (∂y_aca(i, j, k, Δy, f)  - ∂y_aca(i, j-1, k, Δy, f))  / Δy
    @inline ∂z²(i, j, k, ΔzF, ΔzC, f) = (∂z_aac(i, j, k, ΔzF, f) - ∂z_aac(i, j, k-1, ΔzF, f)) / ΔzC[k]

    @inline div_ccc(i, j, k, Δx, Δy, ΔzF, u, v, w) = ∂x_caa(i, j, k, Δx, u) + ∂y_aca(i, j, k, Δy, v) + ∂z_aac(i, j, k, ΔzF, w)

    @inline ∇²(i, j, k, Δx, Δy, ΔzF, ΔzC, f) = ∂x²(i, j, k, Δx, f) + ∂y²(i, j, k, Δy, f) + ∂z²(i, j, k, ΔzF, ΔzC, f)

    #####
    ##### Generate "fake" vertically stretched grid
    #####

    function grid(zF)
        Nz = length(zF) - 1
        ΔzF = [zF[k+1] - zF[k] for k in 1:Nz]
        zC = [(zF[k] + zF[k+1]) / 2 for k in 1:Nz]
        ΔzC = [zC[k+1] - zC[k] for k in 1:Nz-1]
        return zF, zC, ΔzF, ΔzC
    end

    Nz = length(zF) - 1
    zF, zC, ΔzF, ΔzC = grid(zF)

    # Need some halo regions.
    ΔzF = OffsetArray([ΔzF[1], ΔzF...], 0:Nz)
    ΔzC = [ΔzC..., ΔzC[end]]

    # Useful for broadcasting z operations
    ΔzC = reshape(ΔzC, (1, 1, Nz))

    # Temporary hack: Useful for reusing fill_halo_regions! and
    # BatchedTridiagonalSolver which only need Nx, Ny, Nz.
    fake_grid = RegularCartesianGrid(size=(Nx, Ny, Nz), length=(Lx, Ly, Lz))

    #####
    ##### Generate batched tridiagonal system coefficients and solver
    #####

    function λi(Nx, Δx)
        is = reshape(1:Nx, Nx, 1, 1)
        @. (2sin((is-1)*π/Nx) / Δx)^2
    end

    function λj(Ny, Δy)
        js = reshape(1:Ny, 1, Ny, 1)
        @. (2sin((js-1)*π/Ny) / Δy)^2
    end

    kx² = λi(Nx, Δx)
    ky² = λj(Ny, Δy)

    # Lower and upper diagonals are the same
    ld = [1/ΔzF[k] for k in 1:Nz-1]
    ud = copy(ld)

    # Diagonal (different for each i,j)
    @inline δ(k, ΔzF, ΔzC, kx², ky²) = - (1/ΔzF[k-1] + 1/ΔzF[k]) - ΔzC[k] * (kx² + ky²)

    d = zeros(Nx, Ny, Nz)
    for i in 1:Nx, j in 1:Ny
        d[i, j, 1] = -1/ΔzF[1] - ΔzC[1] * (kx²[i] + ky²[j])
        d[i, j, 2:Nz-1] .= [δ(k, ΔzF, ΔzC, kx²[i], ky²[j]) for k in 2:Nz-1]
        d[i, j, Nz] = -1/ΔzF[Nz-1] - ΔzC[Nz] * (kx²[i] + ky²[j])
    end

    #####
    ##### Random right hand side
    #####

    # Random right hand side
    Ru = CellField(Float64, arch, fake_grid, UVelocityBoundaryConditions(fake_grid))
    Rv = CellField(Float64, arch, fake_grid, VVelocityBoundaryConditions(fake_grid))
    Rw = CellField(Float64, arch, fake_grid, WVelocityBoundaryConditions(fake_grid))

    interior(Ru) .= rand(Nx, Ny, Nz)
    interior(Rv) .= rand(Nx, Ny, Nz)
    interior(Rw) .= zeros(Nx, Ny, Nz)

    U = (u=Ru, v=Rv, w=Rw)
    fill_halo_regions!(U, arch)

    _compute_w_from_continuity!(U, fake_grid)

    fill_halo_regions!(Rw, arch)

    R = zeros(Nx, Ny, Nz)
    for i in 1:Nx, j in 1:Ny, k in 1:Nz
        R[i, j, k] = div_ccc(i, j, k, Δx, Δy, ΔzF, Ru.data, Rv.data, Rw.data)
    end

    # @show sum(R)  # should be zero by construction.

    F = zeros(Nx, Ny, Nz)
    F = ΔzC .* R  # RHS needs to be multiplied by ΔzC

    #####
    ##### Solve system
    #####

    F̃ = fft(F, [1, 2])

    btsolver = BatchedTridiagonalSolver(arch, dl=ld, d=d, du=ud, f=F̃, grid=fake_grid)

    ϕ̃ = zeros(Complex{Float64}, Nx, Ny, Nz)
    solve_batched_tridiagonal_system!(ϕ̃, arch, btsolver)

    ϕ = CellField(Float64, arch, fake_grid, PressureBoundaryConditions(fake_grid))
    interior(ϕ) .= real.(ifft(ϕ̃, [1, 2]))
    ϕ.data .= ϕ.data .- mean(interior(ϕ))

    #####
    ##### Compute Laplacian of solution ϕ to test that it's correct
    #####

    fill_halo_regions!(ϕ, arch)

    ∇²ϕ = CellField(Float64, arch, fake_grid, PressureBoundaryConditions(fake_grid))
    for i in 1:Nx, j in 1:Ny, k in 1:Nz
        ∇²ϕ.data[i, j, k] = ∇²(i, j, k, Δx, Δy, ΔzF, ΔzC, ϕ.data)
    end

    return interior(∇²ϕ) ≈ R
end

@testset "Solvers" begin
    @info "Testing Solvers..."

    for arch in archs
        @testset "Batched tridiagonal solver [$arch]" begin
            for Nz in [8, 11, 18]
                @test can_solve_single_tridiagonal_system(arch, Nz)
                @test can_solve_single_tridiagonal_system_with_functions(arch, Nz)
            end

            for Nx in [3, 8], Ny in [5, 16], Nz in [8, 11]
                @test can_solve_batched_tridiagonal_system_with_3D_RHS(arch, Nx, Ny, Nz)
                @test can_solve_batched_tridiagonal_system_with_3D_functions(arch, Nx, Ny, Nz)
            end
        end
    end

    for arch in [CPU()]
        @testset "Vertically stretched Poisson solver [FACR, $arch]" begin
            @info "  Testing vertically stretched Poisson solver [FACR, $arch]..."

            Nx = Ny = 8
            zF = [1, 2, 4, 7, 11, 16, 22, 29, 37]
            @test vertically_stretched_poisson_solver_correct_answer(arch, Nx, Ny, zF)
        end
    end
end
