using Oceananigans.Solvers: solve_poisson_equation!

function ∇²!(grid, f, ∇²f)
    @loop_xyz i j k grid begin
        @inbounds ∇²f[i, j, k] = ∇²(i, j, k, grid, f)
    end
end

function pressure_solver_instantiates(FT, Nx, Ny, Nz, planner_flag)
    grid = RegularCartesianGrid(FT, size=(Nx, Ny, Nz), length=(100, 100, 100))
    solver = PressureSolver(CPU(), grid, PressureBoundaryConditions(grid), planner_flag)
    return true  # Just making sure the PressureSolver does not error/crash.
end

function poisson_ppn_planned_div_free_cpu(FT, Nx, Ny, Nz, planner_flag)
    arch = CPU()
    grid = RegularCartesianGrid(FT, size=(Nx, Ny, Nz), length=(1.0, 2.5, 3.6))
    fbcs = TracerBoundaryConditions(grid)
    pbcs = PressureBoundaryConditions(grid)
    solver = PressureSolver(arch, grid, fbcs)

    RHS = CellField(FT, arch, grid, fbcs)
    interior(RHS) .= rand(Nx, Ny, Nz)
    interior(RHS) .= interior(RHS) .- mean(interior(RHS))

    RHS_orig = deepcopy(RHS)
    solver.storage .= interior(RHS)
    solve_poisson_equation!(solver, grid)

    ϕ   = CellField(FT, arch, grid, pbcs)
    ∇²ϕ = CellField(FT, arch, grid, pbcs)

    interior(ϕ) .= real.(solver.storage)

    fill_halo_regions!(ϕ, arch)
    ∇²!(grid, ϕ, ∇²ϕ)

    fill_halo_regions!(∇²ϕ, arch)

    interior(∇²ϕ) ≈ interior(RHS_orig)
end

function poisson_pnn_planned_div_free_cpu(FT, Nx, Ny, Nz, planner_flag)
    arch = CPU()
    grid = RegularCartesianGrid(FT, size=(Nx, Ny, Nz), length=(1.0, 2.5, 3.6), topology=(Periodic, Bounded, Bounded))
    fbcs = TracerBoundaryConditions(grid)
    pbcs = PressureBoundaryConditions(grid)
    solver = PressureSolver(arch, grid, fbcs)

    RHS = CellField(FT, arch, grid, fbcs)
    interior(RHS) .= rand(Nx, Ny, Nz)
    interior(RHS) .= interior(RHS) .- mean(interior(RHS))

    RHS_orig = deepcopy(RHS)

    solver.storage .= interior(RHS)

    solve_poisson_equation!(solver, grid)

    ϕ   = CellField(FT, arch, grid, pbcs)
    ∇²ϕ = CellField(FT, arch, grid, pbcs)

    interior(ϕ) .= real.(solver.storage)

    fill_halo_regions!(ϕ, arch)
    ∇²!(grid, ϕ, ∇²ϕ)

    fill_halo_regions!(∇²ϕ, arch)

    interior(∇²ϕ) ≈ interior(RHS_orig)
end

function poisson_ppn_planned_div_free_gpu(FT, Nx, Ny, Nz)
    arch = GPU()
    grid = RegularCartesianGrid(FT, size=(Nx, Ny, Nz), length=(1.0, 2.5, 3.6))
    pbcs = PressureBoundaryConditions(grid)
    solver = PressureSolver(arch, grid, pbcs)

    RHS = rand(Nx, Ny, Nz)
    RHS .= RHS .- mean(RHS)
    RHS = CuArray(RHS)

    RHS_orig = copy(RHS)

    solver.storage .= RHS

    # Performing the permutation [a, b, c, d, e, f] -> [a, c, e, f, d, b]
    # in the z-direction in preparation to calculate the DCT in the Poisson
    # solver.
    solver.storage .= cat(solver.storage[:, :, 1:2:Nz], solver.storage[:, :, Nz:-2:2]; dims=3)

    solve_poisson_equation!(solver, grid)

    # Undoing the permutation made above to complete the IDCT.
    solver.storage .= CuArray(reshape(permutedims(cat(solver.storage[:, :, 1:Int(Nz/2)],
                                                      solver.storage[:, :, end:-1:Int(Nz/2)+1]; dims=4), (1, 2, 4, 3)), Nx, Ny, Nz))

    ϕ   = CellField(FT, arch, grid, pbcs)
    ∇²ϕ = CellField(FT, arch, grid, pbcs)

    interior(ϕ) .= real.(solver.storage)

    fill_halo_regions!(ϕ, arch)
    ∇²!(grid, ϕ.data, ∇²ϕ.data)

    fill_halo_regions!(∇²ϕ, arch)
    interior(∇²ϕ) ≈ RHS_orig
end

function poisson_pnn_planned_div_free_gpu(FT, Nx, Ny, Nz)
    arch = GPU()
    grid = RegularCartesianGrid(FT, size=(Nx, Ny, Nz), length=(1.0, 2.5, 3.6), topology=(Periodic, Bounded, Bounded))
    pbcs = PressureBoundaryConditions(grid)
    solver = PressureSolver(arch, grid, pbcs)

    RHS = rand(Nx, Ny, Nz)
    RHS .= RHS .- mean(RHS)
    RHS = CuArray(RHS)

    RHS_orig = copy(RHS)

    storage = solver.storage.storage1
    storage .= RHS

    storage .= cat(storage[:, :, 1:2:Nz], storage[:, :, Nz:-2:2]; dims=3)
    storage .= cat(storage[:, 1:2:Ny, :], storage[:, Ny:-2:2, :]; dims=2)

    solve_poisson_equation!(solver, grid)

    ϕ   = CellField(FT, arch, grid)
    ∇²ϕ = CellField(FT, arch, grid)

    # Indices used when we need views to permuted arrays where the odd indices
    # are iterated over first followed by the even indices.
    p_y_inds = [1:2:Ny..., Ny:-2:2...] |> CuArray
    p_z_inds = [1:2:Nz..., Nz:-2:2...] |> CuArray

    ϕ_p = view(interior(ϕ), 1:Nx, p_y_inds, p_z_inds)

    @. ϕ_p = real(storage)

    fill_halo_regions!(ϕ, arch)
    ∇²!(grid, ϕ.data, ∇²ϕ.data)

    fill_halo_regions!(∇²ϕ, arch)
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
    grid = RegularCartesianGrid(FT, size=(Nx, Ny, Nz), length=(Lx, Ly, Lz))
    solver = PressureSolver(CPU(), grid, TracerBoundaryConditions(grid))

    xC, yC, zC = grid.xC, grid.yC, grid.zC
    xC = reshape(xC, (Nx, 1, 1))
    yC = reshape(yC, (1, Ny, 1))
    zC = reshape(zC, (1, 1, Nz))

    Ψ(x, y, z) = cos(π*mz*z/Lz) * sin(2π*my*y/Ly) * sin(2π*mx*x/Lx)
    f(x, y, z) = -((mz*π/Lz)^2 + (2π*my/Ly)^2 + (2π*mx/Lx)^2) * Ψ(x, y, z)

    @. solver.storage = f(xC, yC, zC)
    solve_poisson_equation!(solver, grid)
    ϕ = real.(solver.storage)

    error = norm(ϕ - Ψ.(xC, yC, zC)) / √(Nx*Ny*Nz)

    @info "Error (ℓ²-norm), $FT, N=($Nx, $Ny, $Nz), m=($mx, $my, $mz): $error"

    isapprox(ϕ, Ψ.(xC, yC, zC), rtol=5e-2)
end

@testset "Pressure solvers" begin
    @info "Testing pressure solvers..."

    @testset "Pressure solver instantiation" begin
        @info "  Testing pressure solver instantiation..."

        for FT in float_types
            @test pressure_solver_instantiates(FT, 32, 32, 32, FFTW.ESTIMATE)
            @test pressure_solver_instantiates(FT, 1,  32, 32, FFTW.ESTIMATE)
            @test pressure_solver_instantiates(FT, 32,  1, 32, FFTW.ESTIMATE)
            @test pressure_solver_instantiates(FT,  1,  1, 32, FFTW.ESTIMATE)
        end
    end

    @testset "Divergence-free solution [CPU]" begin
        @info "  Testing divergence-free solution [CPU]..."

        for N in [7, 10, 16, 20]
            for FT in float_types
                for planner_flag in (FFTW.ESTIMATE, FFTW.MEASURE)
                    @test poisson_ppn_planned_div_free_cpu(FT, N, N, N, planner_flag)
                    @test poisson_ppn_planned_div_free_cpu(FT, 1, N, N, planner_flag)
                    @test poisson_ppn_planned_div_free_cpu(FT, N, 1, N, planner_flag)
                    @test poisson_ppn_planned_div_free_cpu(FT, 1, 1, N, planner_flag)
                end
            end
        end

        Ns = [5, 11, 20, 32]
        for Nx in Ns, Ny in Ns, Nz in Ns, FT in float_types
            @test poisson_ppn_planned_div_free_cpu(FT, Nx, Ny, Nz, FFTW.ESTIMATE)
            @test poisson_pnn_planned_div_free_cpu(FT, Nx, Ny, Nz, FFTW.ESTIMATE)
        end
    end

    @testset "Divergence-free solution [GPU]" begin
        @info "  Testing divergence-free solution [GPU]..."
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
        @info "  Testing analytic solution reconstruction..."
        for N in [32, 48, 64], m in [1, 2, 3]
            @test poisson_ppn_recover_sine_cosine_solution(Float64, N, N, N, 100, 100, 100, m, m, m)
        end
    end
end
