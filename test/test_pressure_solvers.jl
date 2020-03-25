using Oceananigans.Solvers: solve_poisson_equation!

function ∇²!(grid, f, ∇²f)
    @loop_xyz i j k grid begin
        @inbounds ∇²f[i, j, k] = ∇²(i, j, k, grid, f)
    end
end

function pressure_solver_instantiates(FT, Nx, Ny, Nz, planner_flag)
    grid = RegularCartesianGrid(FT, size=(Nx, Ny, Nz), extent=(100, 100, 100))
    solver = PressureSolver(CPU(), grid, PressureBoundaryConditions(grid), planner_flag)
    return true  # Just making sure the PressureSolver does not error/crash.
end

function poisson_ppn_planned_div_free_cpu(FT, Nx, Ny, Nz, planner_flag)
    arch = CPU()
    grid = RegularCartesianGrid(FT, size=(Nx, Ny, Nz), extent=(1.0, 2.5, 3.6))
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
    grid = RegularCartesianGrid(FT, size=(Nx, Ny, Nz), extent=(1.0, 2.5, 3.6), topology=(Periodic, Bounded, Bounded))
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
    grid = RegularCartesianGrid(FT, size=(Nx, Ny, Nz), extent=(1.0, 2.5, 3.6))
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
    grid = RegularCartesianGrid(FT, size=(Nx, Ny, Nz), extent=(1.0, 2.5, 3.6), topology=(Periodic, Bounded, Bounded))
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

#####
##### Test that Poisson solver error converges as error ~ N⁻²
#####

ψ(::Bounded, n, x) = cos(n*x/2)
ψ(::Periodic, n, x) = cos(n*x)

k²(::Bounded, n) = (n/2)^2
k²(::Periodic, n) = n^2

function analytical_poisson_solver_test(arch, N, topo; FT=Float64, mode=1)
    grid = RegularCartesianGrid(FT, topology=topo, size=(N, N, N), x=(0, 2π), y=(0, 2π), z=(0, 2π))
    solver = PressureSolver(arch, grid, TracerBoundaryConditions(grid))

    xC, yC, zC = nodes((Cell, Cell, Cell), grid)

    Tx, Ty, Tz = topology(grid)
    Ψ(x, y, z) = ψ(Tx, mode, x) * ψ(Ty, mode, y) * ψ(Tz, mode, z)
    f(x, y, z) = -(k²(Tx, mode) + k²(Ty, mode) + k²(Tz, mode)) * Ψ(x, y, z)

    @. solver.storage = f(xC, yC, zC)
    solve_poisson_equation!(solver, grid)
    ϕ = real.(solver.storage)

    L¹_error = mean(abs, ϕ - Ψ.(xC, yC, zC))

    return L¹_error
end

function poisson_solver_convergence(arch, topo, N¹, N²; FT=Float64)
    error¹ = analytical_poisson_solver_test(arch, N¹, topo; FT=FT)
    error² = analytical_poisson_solver_test(arch, N², topo; FT=FT)

    rate = log(error¹ / error²) / log(N² / N¹)

    Tx, Ty, Tz = topo
    @info "Convergence of L¹-normed error, $FT, ($(N¹)³ -> $(N²)³), topology=($Tx, $Ty, $Tz): $rate"

    return isapprox(rate, 2, rtol=5e-3)
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

    @hascuda begin
    @testset "Divergence-free solution [GPU]" begin
        @info "  Testing divergence-free solution [GPU]..."
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

    @testset "Convergence to analytical solution" begin
        @info "  Testing convergence to analytical solution..."
        @test_skip poisson_solver_convergence(CPU(), (Periodic, Periodic, Periodic), 2^6, 2^7)
        @test poisson_solver_convergence(CPU(), (Periodic, Periodic, Bounded), 2^6, 2^7)
        @test poisson_solver_convergence(CPU(), (Periodic, Bounded, Bounded), 2^6, 2^7)
        @test poisson_solver_convergence(CPU(), (Bounded, Bounded, Bounded), 2^6, 2^7)
    end
end
