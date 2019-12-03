using LinearAlgebra
using Oceananigans: array_type

function ∇²!(grid, f, ∇²f)
    @loop for k in (1:grid.Nz; (blockIdx().z - 1) * blockDim().z + threadIdx().z)
        @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                @inbounds ∇²f[i, j, k] = ∇²(grid, f, i, j, k)
            end
        end
    end
end

function fftw_planner_works(FT, Nx, Ny, Nz, planner_flag)
    grid = RegularCartesianGrid(FT; size=(Nx, Ny, Nz), length=(100, 100, 100))
    solver = PoissonSolver(CPU(), PPN(), grid)
    true  # Just making sure the PoissonSolver does not error/crash.
end

function poisson_ppn_planned_div_free_cpu(FT, Nx, Ny, Nz, planner_flag)
    arch = CPU()
    grid = RegularCartesianGrid(FT; size=(Nx, Ny, Nz), length=(1.0, 2.5, 3.6))
    solver = PoissonSolver(arch, PPN(), grid)
    fbcs = HorizontallyPeriodicBCs()

    RHS = CellField(FT, arch, grid)
    interior(RHS) .= rand(Nx, Ny, Nz)
    interior(RHS) .= interior(RHS) .- mean(interior(RHS))

    RHS_orig = deepcopy(RHS)
    solver.storage .= interior(RHS)
    solve_poisson_3d!(solver, grid)

    ϕ   = CellField(FT, arch, grid)
    ∇²ϕ = CellField(FT, arch, grid)

    interior(ϕ) .= real.(solver.storage)

    fill_halo_regions!(ϕ.data, fbcs, arch, grid)
    ∇²!(grid, ϕ, ∇²ϕ)

    fill_halo_regions!(∇²ϕ.data, fbcs, arch, grid)

    interior(∇²ϕ) ≈ interior(RHS_orig)
end

function poisson_pnn_planned_div_free_cpu(FT, Nx, Ny, Nz, planner_flag)
    arch = CPU()
    grid = RegularCartesianGrid(FT; size=(Nx, Ny, Nz), length=(1.0, 2.5, 3.6))
    solver = PoissonSolver(arch, PNN(), grid)
    fbcs = ChannelBCs()

    RHS = CellField(FT, arch, grid)
    interior(RHS) .= rand(Nx, Ny, Nz)
    interior(RHS) .= interior(RHS) .- mean(interior(RHS))

    RHS_orig = deepcopy(RHS)

    solver.storage .= interior(RHS)

    solve_poisson_3d!(solver, grid)

    ϕ   = CellField(FT, arch, grid)
    ∇²ϕ = CellField(FT, arch, grid)

    interior(ϕ) .= real.(solver.storage)

    fill_halo_regions!(ϕ.data, fbcs, arch, grid)
    ∇²!(grid, ϕ, ∇²ϕ)

    fill_halo_regions!(∇²ϕ.data, fbcs, arch, grid)

    interior(∇²ϕ) ≈ interior(RHS_orig)
end

function poisson_ppn_planned_div_free_gpu(FT, Nx, Ny, Nz)
    arch = GPU()
    grid = RegularCartesianGrid(FT; size=(Nx, Ny, Nz), length=(1.0, 2.5, 3.6))
    solver = PoissonSolver(arch, PPN(), grid)
    fbcs = HorizontallyPeriodicBCs()

    RHS = rand(Nx, Ny, Nz)
    RHS .= RHS .- mean(RHS)
    RHS = CuArray(RHS)

    RHS_orig = copy(RHS)

    solver.storage .= RHS

    # Performing the permutation [a, b, c, d, e, f] -> [a, c, e, f, d, b]
    # in the z-direction in preparation to calculate the DCT in the Poisson
    # solver.
    solver.storage .= cat(solver.storage[:, :, 1:2:Nz], solver.storage[:, :, Nz:-2:2]; dims=3)

    solve_poisson_3d!(solver, grid)

    # Undoing the permutation made above to complete the IDCT.
    solver.storage .= CuArray(reshape(permutedims(cat(solver.storage[:, :, 1:Int(Nz/2)],
                                                      solver.storage[:, :, end:-1:Int(Nz/2)+1]; dims=4), (1, 2, 4, 3)), Nx, Ny, Nz))

    ϕ   = CellField(FT, arch, grid)
    ∇²ϕ = CellField(FT, arch, grid)

    interior(ϕ) .= real.(solver.storage)

    fill_halo_regions!(ϕ.data, fbcs, arch, grid)
    ∇²!(grid, ϕ.data, ∇²ϕ.data)

    fill_halo_regions!(∇²ϕ.data, fbcs, arch, grid)
    interior(∇²ϕ) ≈ RHS_orig
end

function poisson_pnn_planned_div_free_gpu(FT, Nx, Ny, Nz)
    arch = GPU()
    grid = RegularCartesianGrid(FT; size=(Nx, Ny, Nz), length=(1.0, 2.5, 3.6))
    solver = PoissonSolver(arch, PNN(), grid)
    fbcs = ChannelBCs()

    RHS = rand(Nx, Ny, Nz)
    RHS .= RHS .- mean(RHS)
    RHS = CuArray(RHS)

    RHS_orig = copy(RHS)

    solver.storage .= RHS

    solver.storage .= cat(solver.storage[:, :, 1:2:Nz], solver.storage[:, :, Nz:-2:2]; dims=3)
    solver.storage .= cat(solver.storage[:, 1:2:Ny, :], solver.storage[:, Ny:-2:2, :]; dims=2)

    solve_poisson_3d!(solver, grid)

    ϕ   = CellField(FT, arch, grid)
    ∇²ϕ = CellField(FT, arch, grid)

    ϕ_p = view(interior(ϕ), 1:Nx, solver.p_y_inds, solver.p_z_inds)

    @. ϕ_p = real(solver.storage)

    fill_halo_regions!(ϕ.data, fbcs, arch, grid)
    ∇²!(grid, ϕ.data, ∇²ϕ.data)

    fill_halo_regions!(∇²ϕ.data, fbcs, arch, grid)
    interior(∇²ϕ) ≈ RHS_orig
end

"""
    poisson_ppn_recover_sine_cosine_solution(FT, Nx, Ny, Nz, Lx, Ly, Lz, mx, my, mz)

Test that the Poisson solver can recover an analytic solution. In this test, we
are trying to see if the solver can recover the solution

    ``\\Psi(x, y, z) = cos(\\pi m_z z / L_z) sin(2\\pi m_y y / L_y) sin(2\\pi m_x x / L_x)``

by giving it the source term or right hand side (RHS), which is

    ``f(x, y, z) = \\nabla^2 \\Psi(x, y, z) =
    -((\\pi m_z / L_z)^2 + (2\\pi m_y / L_y)^2 + (2\\pi m_x/L_x)^2) \\Psi(x, y, z)``.
"""
function poisson_ppn_recover_sine_cosine_solution(FT, Nx, Ny, Nz, Lx, Ly, Lz, mx, my, mz)
    grid = RegularCartesianGrid(FT; size=(Nx, Ny, Nz), length=(Lx, Ly, Lz))
    solver = PoissonSolver(CPU(), PPN(), grid)

    xC, yC, zC = grid.xC, grid.yC, grid.zC
    xC = reshape(xC, (Nx, 1, 1))
    yC = reshape(yC, (1, Ny, 1))
    zC = reshape(zC, (1, 1, Nz))

    Ψ(x, y, z) = cos(π*mz*z/Lz) * sin(2π*my*y/Ly) * sin(2π*mx*x/Lx)
    f(x, y, z) = -((mz*π/Lz)^2 + (2π*my/Ly)^2 + (2π*mx/Lx)^2) * Ψ(x, y, z)

    @. solver.storage = f(xC, yC, zC)
    solve_poisson_3d!(solver, grid)
    ϕ = real.(solver.storage)

    error = norm(ϕ - Ψ.(xC, yC, zC)) / √(Nx*Ny*Nz)

    @info "Error (ℓ²-norm), $FT, N=($Nx, $Ny, $Nz), m=($mx, $my, $mz): $error"

    isapprox(ϕ, Ψ.(xC, yC, zC); rtol=5e-2)
end

function can_solve_single_tridiagonal_system(arch, N)
    ArrayType = array_type(arch)

    a = rand(N-1) |> ArrayType
    b = 3 .+ rand(N) |> ArrayType  # +3 to ensure diagonal dominance.
    c = rand(N-1) |> ArrayType
    f = rand(N) |> ArrayType

    M = Tridiagonal(a, b, c)
    ϕ_correct = M \ f  # This solve probably invokes scalar CuArray operations on the GPU!

    ϕ = reshape(zeros(N), (1, 1, N)) |> ArrayType

    grid = RegularCartesianGrid(size=(1, 1, N), length=(1, 1, 1))
    btsolver = BatchedTridiagonalSolver(arch; dl=a, d=b, du=c, f=f, grid=grid)

    solve_batched_tridiagonal_system!(ϕ, arch, btsolver)

    return ϕ[:] ≈ ϕ_correct
end

function can_solve_single_tridiagonal_system_with_functions(arch, N)
    ArrayType = array_type(arch)

    grid = RegularCartesianGrid(size=(1, 1, N), length=(1, 1, 1))

    a = rand(N-1) |> ArrayType
    c = rand(N-1) |> ArrayType

    @inline b(i, j, k, grid, p) = 3 .+ cos(2π*grid.zC[k])  # +3 to ensure diagonal dominance.
    @inline f(i, j, k, grid, p) = sin(2π*grid.zC[k])

    bₐ = [b(1, 1, k, grid, nothing) for k in 1:N] |> ArrayType
    fₐ = [f(1, 1, k, grid, nothing) for k in 1:N] |> ArrayType

    M = Tridiagonal(a, bₐ, c)
    ϕ_correct = M \ fₐ  # This solve probably invokes scalar CuArray operations on the GPU!

    ϕ = reshape(zeros(N), (1, 1, N)) |> ArrayType

    btsolver = BatchedTridiagonalSolver(arch; dl=a, d=b, du=c, f=f, grid=grid)

    solve_batched_tridiagonal_system!(ϕ, arch, btsolver)

    return ϕ[:] ≈ ϕ_correct
end

function can_solve_batched_tridiagonal_system_with_3D_RHS(arch, Nx, Ny, Nz)
    ArrayType = array_type(arch)

    a = rand(Nz-1) |> ArrayType
    b = 3 .+ rand(Nz) |> ArrayType  # +3 to ensure diagonal dominance.
    c = rand(Nz-1) |> ArrayType
    f = rand(Nx, Ny, Nz) |> ArrayType

    M = Tridiagonal(a, b, c)
    ϕ_correct = zeros(Nx, Ny, Nz) |> ArrayType

    for i = 1:Nx, j = 1:Ny
        ϕ_correct[i, j, :] .= M \ f[i, j, :]
    end

    grid = RegularCartesianGrid(size=(Nx, Ny, Nz), length=(1, 1, 1))
    btsolver = BatchedTridiagonalSolver(arch; dl=a, d=b, du=c, f=f, grid=grid)

    ϕ = zeros(Nx, Ny, Nz) |> ArrayType

    solve_batched_tridiagonal_system!(ϕ, arch, btsolver)

    return ϕ ≈ ϕ_correct
end

function can_solve_batched_tridiagonal_system_with_3D_functions(arch, Nx, Ny, Nz)
    ArrayType = array_type(arch)

    grid = RegularCartesianGrid(size=(Nx, Ny, Nz), length=(1, 1, 1))

    a = rand(Nz-1) |> ArrayType
    c = rand(Nz-1) |> ArrayType

    @inline b(i, j, k, grid, p) = 3 + grid.xC[i]*grid.yC[j] * cos(2π*grid.zC[k])
    @inline f(i, j, k, grid, p) = (grid.xC[i] + grid.yC[j]) * sin(2π*grid.zC[k])

    ϕ_correct = zeros(Nx, Ny, Nz) |> ArrayType

    for i = 1:Nx, j = 1:Ny
        bₐ = [b(i, j, k, grid, nothing) for k in 1:Nz] |> ArrayType
        M = Tridiagonal(a, bₐ, c)

        fₐ = [f(i, j, k, grid, nothing) for k in 1:Nz] |> ArrayType
        ϕ_correct[i, j, :] .= M \ fₐ
    end

    btsolver = BatchedTridiagonalSolver(arch; dl=a, d=b, du=c, f=f, grid=grid)

    ϕ = zeros(Nx, Ny, Nz) |> ArrayType
    solve_batched_tridiagonal_system!(ϕ, arch, btsolver)

    return ϕ ≈ ϕ_correct
end

@testset "Solvers" begin
    println("Testing Solvers...")

    @testset "FFTW plans" begin
        println("  Testing FFTW planning...")

        for FT in float_types
            @test fftw_planner_works(FT, 32, 32, 32, FFTW.ESTIMATE)
            @test fftw_planner_works(FT, 1,  32, 32, FFTW.ESTIMATE)
            @test fftw_planner_works(FT, 32,  1, 32, FFTW.ESTIMATE)
            @test fftw_planner_works(FT,  1,  1, 32, FFTW.ESTIMATE)
        end
    end

    @testset "Divergence-free solution [CPU]" begin
        println("  Testing divergence-free solution [CPU]...")

        for N in [7, 10, 16, 20]
            for FT in float_types
                @test poisson_ppn_planned_div_free_cpu(FT, 1, N, N, FFTW.ESTIMATE)
                @test poisson_ppn_planned_div_free_cpu(FT, N, 1, N, FFTW.ESTIMATE)
                @test poisson_ppn_planned_div_free_cpu(FT, 1, 1, N, FFTW.ESTIMATE)

                @test poisson_pnn_planned_div_free_cpu(FT, 1, N, N, FFTW.ESTIMATE)

                # Commented because https://github.com/climate-machine/Oceananigans.jl/issues/99
                # for planner_flag in [FFTW.ESTIMATE, FFTW.MEASURE]
                #     @test test_3d_poisson_ppn_planned!_div_free(mm, N, N, N, planner_flag)
                #     @test test_3d_poisson_ppn_planned!_div_free(mm, 1, N, N, planner_flag)
                #     @test test_3d_poisson_ppn_planned!_div_free(mm, N, 1, N, planner_flag)
                # end
            end
        end

        Ns = [5, 11, 20, 32]
        for Nx in Ns, Ny in Ns, Nz in Ns, FT in float_types
            @test poisson_ppn_planned_div_free_cpu(FT, Nx, Ny, Nz, FFTW.ESTIMATE)
            @test poisson_pnn_planned_div_free_cpu(FT, Nx, Ny, Nz, FFTW.ESTIMATE)
        end
    end

    @testset "Divergence-free solution [GPU]" begin
        println("  Testing divergence-free solution [GPU]...")
        @hascuda begin
            for FT in [Float64]
                @test poisson_ppn_planned_div_free_gpu(FT, 16, 16, 16)
                @test poisson_ppn_planned_div_free_gpu(FT, 32, 32, 32)
                @test poisson_ppn_planned_div_free_gpu(FT, 32, 32, 16)
                @test poisson_ppn_planned_div_free_gpu(FT, 16, 32, 24)

                @test poisson_pnn_planned_div_free_gpu(FT, 16, 16, 16)
                @test poisson_pnn_planned_div_free_gpu(FT, 32, 32, 32)
                @test poisson_pnn_planned_div_free_gpu(FT, 32, 32, 16)
                @test poisson_pnn_planned_div_free_gpu(FT, 16, 32, 24)
            end
        end
    end

    @testset "Analytic solution reconstruction" begin
        println("  Testing analytic solution reconstruction...")
        for N in [32, 48, 64], m in [1, 2, 3]
            @test poisson_ppn_recover_sine_cosine_solution(Float64, N, N, N, 100, 100, 100, m, m, m)
        end
    end

    @testset "Batched tridiagonal solver [CPU]" begin
        arch = CPU()
        for Nz in [8, 11, 18]
            @test can_solve_single_tridiagonal_system(arch, Nz)
            @test can_solve_single_tridiagonal_system_with_functions(arch, Nz)
        end

        for Nx in [3, 8], Ny in [5, 16], Nz in [8, 11, 18]
            @test can_solve_batched_tridiagonal_system_with_3D_RHS(arch, Nx, Ny, Nz)
            @test can_solve_batched_tridiagonal_system_with_3D_functions(arch, Nx, Ny, Nz)
        end
    end

    @testset "Batched tridiagonal solver [GPU]" begin
        arch = GPU()
        for Nz in [8, 11, 18]
            @test can_solve_single_tridiagonal_system(arch, Nz)
            @test can_solve_single_tridiagonal_system_with_functions(arch, Nz)
        end

        for Nx in [16, 32], Ny in [16, 32], Nz in [11, 16]
            @test can_solve_batched_tridiagonal_system_with_3D_RHS(arch, Nx, Ny, Nz)
            @test can_solve_batched_tridiagonal_system_with_3D_functions(arch, Nx, Ny, Nz)
        end
    end
end
