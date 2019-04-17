using Test
using FFTW
using Statistics: mean
using GPUifyLoops: @launch, @loop, @synchronize

using Oceananigans

# Increment and decrement integer a with periodic wrapping. So if n == 10 then
# incmod1(11, n) = 1 and decmod1(0, n) = 10.
@inline incmod1(a, n) = ifelse(a==n, 1, a + 1)
@inline decmod1(a, n) = ifelse(a==1, n, a - 1)

@inline δx_c2f(g::RegularCartesianGrid, f, i, j, k) = @inbounds f[i, j, k] - f[decmod1(i, g.Nx), j, k]
@inline δy_c2f(g::RegularCartesianGrid, f, i, j, k) = @inbounds f[i, j, k] - f[i, decmod1(j, g.Ny), k]

@inline function δz_c2f(g::RegularCartesianGrid, f, i, j, k)
    if k == 1
        return 0
    else
        @inbounds return f[i, j, k-1] - f[i, j, k]
    end
end

@inline δx²_c2f2c(g::RegularCartesianGrid, f, i, j, k) = δx_c2f(g, f, incmod1(i, g.Nx), j, k) - δx_c2f(g, f, i, j, k)
@inline δy²_c2f2c(g::RegularCartesianGrid, f, i, j, k) = δy_c2f(g, f, i, incmod1(j, g.Ny), k) - δy_c2f(g, f, i, j, k)

@inline function δz²_c2f2c(g::RegularCartesianGrid, f, i, j, k)
    if k == g.Nz
        return δz_c2f(g, f, i, j, k)
    else
        return δz_c2f(g, f, i, j, k) - δz_c2f(g, f, i, j, k+1)
    end
end

@inline function ∇²_ppn(g::RegularCartesianGrid, f, i, j, k)
	(δx²_c2f2c(g, f, i, j, k) / g.Δx^2) + (δy²_c2f2c(g, f, i, j, k) / g.Δy^2) + (δz²_c2f2c(g, f, i, j, k) / g.Δz^2)
end

function ∇²_ppn!(grid::RegularCartesianGrid, f, ∇²f)
    @loop for k in (1:grid.Nz; blockIdx().z)
        @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                @inbounds ∇²f[i, j, k] = ∇²_ppn(grid, f, i, j, k)
            end
        end
    end

    @synchronize
end

function solve_poisson_3d_ppn!(g::RegularCartesianGrid, f::CellField, ϕ::CellField)
    kx² = zeros(g.Nx, 1)
    ky² = zeros(g.Ny, 1)
    kz² = zeros(g.Nz, 1)

    for i in 1:g.Nx; kx²[i] = (2sin((i-1)*π/g.Nx)    / (g.Lx/g.Nx))^2; end
    for j in 1:g.Ny; ky²[j] = (2sin((j-1)*π/g.Ny)    / (g.Ly/g.Ny))^2; end
    for k in 1:g.Nz; kz²[k] = (2sin((k-1)*π/(2g.Nz)) / (g.Lz/g.Nz))^2; end

    FFTW.r2r!(f.data, FFTW.REDFT10, 3)
    FFTW.fft!(f.data, [1, 2])

    for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
        @inbounds ϕ.data[i, j, k] = -f.data[i, j, k] / (kx²[i] + ky²[j] + kz²[k])
    end
    ϕ.data[1, 1, 1] = 0

    FFTW.ifft!(ϕ.data, [1, 2])

    @. ϕ.data = real(ϕ.data) / (2g.Nz)

    FFTW.r2r!(ϕ.data, FFTW.REDFT01, 3)

    nothing
end

function poisson_ppn_planned_div_free_cpu(ft, Nx, Ny, Nz, planner_flag)
    g = RegularCartesianGrid(ft, (Nx, Ny, Nz), (100, 100, 100))

    RHS = CellField(Complex{ft}, CPU(), g)
    RHS_orig = CellField(Complex{ft}, CPU(), g)
    ϕ = CellField(Complex{ft}, CPU(), g)
    ∇²ϕ = CellField(Complex{ft}, CPU(), g)

    RHS.data .= rand(Nx, Ny, Nz)
    RHS.data .= RHS.data .- mean(RHS.data)

    RHS_orig.data .= copy(RHS.data)

    solve_poisson_3d_ppn!(g, RHS, ϕ)
    ∇²_ppn!(g, ϕ, ∇²ϕ)

    ∇²ϕ.data ≈ RHS_orig.data
end

@test poisson_ppn_planned_div_free_cpu(Float64, 16, 16, 16, FFTW.ESTIMATE)
