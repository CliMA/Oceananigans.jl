using LinearAlgebra
using Oceananigans.Architectures: array_type

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

    grid = RegularCartesianGrid(size=(1, 1, N), extent=(1, 1, 1))
    btsolver = BatchedTridiagonalSolver(arch; dl=a, d=b, du=c, f=f, grid=grid)

    solve_batched_tridiagonal_system!(ϕ, arch, btsolver)

    return Array(ϕ[:]) ≈ ϕ_correct
end

function can_solve_single_tridiagonal_system_with_functions(arch, N)
    ArrayType = array_type(arch)

    grid = RegularCartesianGrid(size=(1, 1, N), extent=(1, 1, 1))

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

    grid = RegularCartesianGrid(size=(Nx, Ny, Nz), extent=(1, 1, 1))
    btsolver = BatchedTridiagonalSolver(arch; dl=a, d=b, du=c, f=f, grid=grid)

    ϕ = zeros(Nx, Ny, Nz) |> ArrayType

    solve_batched_tridiagonal_system!(ϕ, arch, btsolver)

    return Array(ϕ) ≈ ϕ_correct
end

function can_solve_batched_tridiagonal_system_with_3D_functions(arch, Nx, Ny, Nz)
    ArrayType = array_type(arch)

    grid = RegularCartesianGrid(size=(Nx, Ny, Nz), extent=(1, 1, 1))

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
    Nz = length(zF) - 1
    vs_grid = VerticallyStretchedCartesianGrid(size=(Nx, Ny, Nz), x=(0, 1), y=(0, 1), zF=zF)

    ΔzC = vs_grid.ΔzC
    ΔzF = vs_grid.ΔzF

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

    kx² = λi(Nx, vs_grid.Δx)
    ky² = λj(Ny, vs_grid.Δy)

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
    Ru = CenterField(Float64, arch, vs_grid, UVelocityBoundaryConditions(vs_grid))
    Rv = CenterField(Float64, arch, vs_grid, VVelocityBoundaryConditions(vs_grid))
    Rw = CenterField(Float64, arch, vs_grid, WVelocityBoundaryConditions(vs_grid))

    interior(Ru) .= rand(Nx, Ny, Nz)
    interior(Rv) .= rand(Nx, Ny, Nz)
    interior(Rw) .= zeros(Nx, Ny, Nz)

    U = (u=Ru, v=Rv, w=Rw)
    fill_halo_regions!(U, arch, nothing, nothing)

    # _compute_w_from_continuity!(U, grid)
    # Rw[i, j, 1] = 0 will be enforced via halo regions.
    for i in 1:Nx, j in 1:Ny, k in 2:Nz
        @inbounds Rw[i, j, k] = Rw[i, j, k-1] - ΔzC[k] * hdivᶜᶜᵃ(i, j, k, vs_grid, Ru, Rv)
    end

    fill_halo_regions!(Rw, arch, nothing, nothing)

    R = zeros(Nx, Ny, Nz)
    for i in 1:Nx, j in 1:Ny, k in 1:Nz
        R[i, j, k] = divᶜᶜᶜ(i, j, k, vs_grid, Ru, Rv, Rw) #  div_ccc(i, j, k, Δx, Δy, ΔzF, Ru.data, Rv.data, Rw.data)
    end

    F = reshape(ΔzC[1:Nz], 1, 1, Nz) .* R  # RHS needs to be multiplied by ΔzC

    #####
    ##### Solve system
    #####

    F̃ = fft(F, [1, 2])

    btsolver = BatchedTridiagonalSolver(arch, dl=ld, d=d, du=ud, f=F̃, grid=vs_grid)

    ϕ̃ = zeros(Complex{Float64}, Nx, Ny, Nz)
    solve_batched_tridiagonal_system!(ϕ̃, arch, btsolver)

    ϕ = CenterField(Float64, arch, vs_grid, PressureBoundaryConditions(vs_grid))
    interior(ϕ) .= real.(ifft(ϕ̃, [1, 2]))
    ϕ.data .= ϕ.data .- mean(interior(ϕ))

    #####
    ##### Compute Laplacian of solution ϕ to test that it's correct
    #####

    fill_halo_regions!(ϕ, arch, nothing, nothing)

    ∇²ϕ = CenterField(Float64, arch, vs_grid, PressureBoundaryConditions(vs_grid))
    for i in 1:Nx, j in 1:Ny, k in 1:Nz
        ∇²ϕ.data[i, j, k] = Oceananigans.Operators.∇²(i, j, k, vs_grid, ϕ.data)
    end

    return interior(∇²ϕ) ≈ R
end

@testset "Solvers" begin
    @info "Testing Solvers..."

    # for arch in archs
    #     @testset "Batched tridiagonal solver [$arch]" begin
    #         for Nz in [8, 11, 18]
    #             @test can_solve_single_tridiagonal_system(arch, Nz)
    #             @test can_solve_single_tridiagonal_system_with_functions(arch, Nz)
    #         end

    #         for Nx in [3, 8], Ny in [5, 16], Nz in [8, 11]
    #             @test can_solve_batched_tridiagonal_system_with_3D_RHS(arch, Nx, Ny, Nz)
    #             @test can_solve_batched_tridiagonal_system_with_3D_functions(arch, Nx, Ny, Nz)
    #         end
    #     end
    # end

    for arch in [CPU()]
        @testset "Vertically stretched Poisson solver [FACR, $arch]" begin
            @info "  Testing vertically stretched Poisson solver [FACR, $arch]..."

            Nx = Ny = 8
            zF = [1, 2, 4, 7, 11, 16, 22, 29, 37]
            @test vertically_stretched_poisson_solver_correct_answer(arch, Nx, Ny, zF)
        end
    end
end
