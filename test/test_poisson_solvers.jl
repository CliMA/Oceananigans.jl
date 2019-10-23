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

@testset "Poisson solvers" begin
    println("Testing Poisson solvers...")

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
end
