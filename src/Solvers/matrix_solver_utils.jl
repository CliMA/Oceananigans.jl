using Oceananigans.Architectures
using Oceananigans.Architectures: device, device_event
import Oceananigans.Architectures: architecture, unified_array
using CUDA, CUDA.CUSPARSE
using KernelAbstractions: @kernel, @index

using LinearAlgebra, SparseArrays, IncompleteLU
using SparseArrays: fkeep!

# Utils for sparse matrix manipulation

@inline constructors(::CPU, A::SparseMatrixCSC) = (A.m, A.n, A.colptr, A.rowval, A.nzval)
@inline constructors(::GPU, A::SparseMatrixCSC) = (CuArray(A.colptr), CuArray(A.rowval), CuArray(A.nzval),  (A.m, A.n))
@inline constructors(::CPU, A::CuSparseMatrixCSC) = (A.dims[1], A.dims[2], Int64.(Array(A.colPtr)), Int64.(Array(A.rowVal)), Array(A.nzVal))
@inline constructors(::GPU, A::CuSparseMatrixCSC) = (A.colPtr, A.rowVal, A.nzVal,  A.dims)
@inline constructors(::CPU, m::Number, n::Number, constr::Tuple) = (m, n, constr...)
@inline constructors(::GPU, m::Number, n::Number, constr::Tuple) = (constr..., (m, n))

@inline unpack_constructors(::CPU, constr::Tuple) = (constr[3], constr[4], constr[5])
@inline unpack_constructors(::GPU, constr::Tuple) = (constr[1], constr[2], constr[3])
@inline copy_unpack_constructors(::CPU, constr::Tuple) = deepcopy((constr[3], constr[4], constr[5]))
@inline copy_unpack_constructors(::GPU, constr::Tuple) = deepcopy((constr[1], constr[2], constr[3]))

@inline arch_sparse_matrix(::CPU, constr::Tuple) = SparseMatrixCSC(constr...)
@inline arch_sparse_matrix(::GPU, constr::Tuple) = CuSparseMatrixCSC(constr...)
@inline arch_sparse_matrix(::CPU, A::CuSparseMatrixCSC)   = SparseMatrixCSC(constructors(CPU(), A)...)
@inline arch_sparse_matrix(::GPU, A::SparseMatrixCSC)     = CuSparseMatrixCSC(constructors(GPU(), A)...)

@inline arch_sparse_matrix(::CPU, A::SparseMatrixCSC)   = A
@inline arch_sparse_matrix(::GPU, A::CuSparseMatrixCSC) = A

# We need to update the diagonal element each time the time step changes!!
function update_diag!(constr, arch, M, N, diag, Δt, disp)   
    colptr, rowval, nzval = unpack_constructors(arch, constr)
    loop! = _update_diag!(device(arch), min(256, M), M)
    event = loop!(nzval, colptr, rowval, diag, Δt, disp; dependencies=device_event(arch))
    wait(device(arch), event)

    constr = constructors(arch, M, N, (colptr, rowval, nzval))
end

@kernel function _update_diag!(nzval, colptr, rowval, diag, Δt, disp)
    col = @index(Global, Linear)
    col = col + disp
    map = 1
    for idx in colptr[col]:colptr[col+1] - 1
       if rowval[idx] + disp == col 
           map = idx 
            break
        end
    end
    nzval[map] += diag[col - disp] / Δt^2 
end

@kernel function _get_inv_diag!(invdiag, colptr, rowval, nzval)
    col = @index(Global, Linear)
    map = 1
    for idx in colptr[col]:colptr[col+1] - 1
        if rowval[idx] == col
            map = idx 
            break
        end
    end
    if nzval[map] == 0
        invdiag[col] = 0 
    else
        invdiag[col] = 1 / nzval[map]
    end
end

@kernel function _get_diag!(diag, colptr, rowval, nzval)
    col = @index(Global, Linear)
    map = 1
    for idx in colptr[col]:colptr[col+1] - 1
        if rowval[idx] == col
            map = idx 
            break
        end
    end
    diag[col] = nzval[map]
end

#unfortunately this cannot run on a GPU so we have to resort to that ugly loop in _update_diag!
@inline map_row_to_diag_element(i, rowval, colptr) =  colptr[i] - 1 + findfirst(rowval[colptr[i]:colptr[i+1]-1] .== i)

@inline function validate_laplacian_direction(N, topo, reduced_dim)  
    dim = N > 1 && reduced_dim == false
    if N < 3 && topo == Bounded && dim == true
        throw(ArgumentError("Cannot calculate Laplacian in bounded domain with N < 3!"))
    end

    return dim
end

@inline validate_laplacian_size(N, dim) = dim == true ? N : 1
  
@inline ensure_diagonal_elements_are_present!(A) = fkeep!(A, (i, j, x) -> (i == j || !iszero(x)))

"""
    compute_matrix_for_linear_operation(arch, template_field, linear_operation!, args...;
                                        boundary_conditions_input=nothing,
                                        boundary_conditions_output=nothing)

Return the sparse matrix that corresponds to the `linear_operation!`. The `linear_operation!`
is expected to have the argument structure:

```julia
linear_operation!(output, input, args...)
```

Keyword arguments `boundary_conditions_input` and `boundary_conditions_output` determine the
boundary conditions that the `input` and `output` fields are expected to have. If `nothing`
is provided, then the `input` and `output` fields inherit the default boundary conditions
according to the `template_field`.

For `architecture = CPU()` the matrix returned is a `SparseArrays.SparseMatrixCSC`; for `GPU()`
is a `CUDA.CuSparseMatrixCSC`.
"""
function compute_matrix_for_linear_operation(::CPU, template_field, linear_operation!, args...;
                                             boundary_conditions_input=nothing,
                                             boundary_conditions_output=nothing)

    loc = location(template_field)
    Nx, Ny, Nz = size(template_field)
    grid = template_field.grid

    # allocate matrix A
    A = spzeros(eltype(grid), Nx*Ny*Nz, Nx*Ny*Nz)

    if boundary_conditions_input === nothing
        boundary_conditions_input = FieldBoundaryConditions(grid, loc, template_field.indices)
    end

    if boundary_conditions_output === nothing
        boundary_conditions_output = FieldBoundaryConditions(grid, loc, template_field.indices)
    end

    # allocate fields eᵢⱼₖ and Aeᵢⱼₖ = A*eᵢⱼₖ
     eᵢⱼₖ = Field(loc, grid; boundary_conditions=boundary_conditions_input)
    Aeᵢⱼₖ = Field(loc, grid; boundary_conditions=boundary_conditions_output)

    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        parent(eᵢⱼₖ)  .= 0
        parent(Aeᵢⱼₖ) .= 0

        eᵢⱼₖ[i, j, k] = 1

        fill_halo_regions!(eᵢⱼₖ)

        linear_operation!(Aeᵢⱼₖ, eᵢⱼₖ, args...)

        A[:, Ny*Nx*(k-1) + Nx*(j-1) + i] .= vec(Aeᵢⱼₖ)
    end

    return A
end

function compute_matrix_for_linear_operation(::GPU, template_field, linear_operation!, args...;
                                             boundary_conditions_input=nothing,
                                             boundary_conditions_output=nothing)

    loc = location(template_field)
    Nx, Ny, Nz = size(template_field)
    grid = template_field.grid

    if boundary_conditions_input === nothing
        boundary_conditions_input = FieldBoundaryConditions(grid, loc, template_field.indices)
    end

    if boundary_conditions_output === nothing
        boundary_conditions_output = FieldBoundaryConditions(grid, loc, template_field.indices)
    end

    # allocate fields eᵢⱼₖ and Aeᵢⱼₖ = A*eᵢⱼₖ; A is the matrix to be computed
     eᵢⱼₖ = Field(loc, grid; boundary_conditions=boundary_conditions_input)
    Aeᵢⱼₖ = Field(loc, grid; boundary_conditions=boundary_conditions_output)

    colptr = CuArray{Int}(undef, Nx*Ny*Nz + 1)
    rowval = CuArray{Int}(undef, 0)
    nzval  = CuArray{eltype(grid)}(undef, 0)

    CUDA.@allowscalar colptr[1] = 1

    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        parent(eᵢⱼₖ)  .= 0
        parent(Aeᵢⱼₖ) .= 0

        CUDA.@allowscalar eᵢⱼₖ[i, j, k] = 1

        fill_halo_regions!(eᵢⱼₖ)

        linear_operation!(Aeᵢⱼₖ, eᵢⱼₖ, args...)

        count = 0
        for n in 1:Nz, m in 1:Ny, l in 1:Nx
            Aeᵢⱼₖₗₘₙ = CUDA.@allowscalar Aeᵢⱼₖ[l, m, n]
            if Aeᵢⱼₖₗₘₙ != 0
                append!(rowval, Ny*Nx*(n-1) + Nx*(m-1) + l)
                append!(nzval, Aeᵢⱼₖₗₘₙ)
                count += 1
            end
        end

        CUDA.@allowscalar colptr[Ny*Nx*(k-1) + Nx*(j-1) + i + 1] = colptr[Ny*Nx*(k-1) + Nx*(j-1) + i] + count
    end

    return CuSparseMatrixCSC(colptr, rowval, nzval, (Nx*Ny*Nz, Nx*Ny*Nz))
end
