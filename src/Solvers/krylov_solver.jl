import Krylov
import Krylov.FloatOrComplex

## Wrapper for AbstractField so that it behaves as a vector for Krylov.jl
struct KrylovField{T, F <: AbstractField} <: AbstractVector{T}
    field::F
end

function KrylovField(field::F) where F <: AbstractField
    T = eltype(field)
    return KrylovField{T,F}(field)
end

function Base.similar(kf::KrylovField)
    field = similar(kf.field)
    KrylovField(field)
end

function Base.isempty(kf::KrylovField)
    bool = isempty(kf.field)
    return bool
end

Base.size(kf::KrylovField) = size(kf.field)
Base.length(kf::KrylovField) = length(kf.field)
Base.getindex(kf::KrylovField, i::Int) = getindex(kf.field, i)

## Internal methods of Krylov.jl that need to be overloaded
function Krylov.kscal!(n::Integer, s::T, x::KrylovField{T}) where T <: FloatOrComplex
    xp = parent(x.field)
    xp .*= s
    return x
end

function Krylov.kdiv!(n::Integer, x::KrylovField{T}, s::T) where T <: FloatOrComplex
    xp = parent(x.field)
    xp ./= s
    return x
end

function Krylov.kaxpy!(n::Integer, s::T, x::KrylovField{T}, y::KrylovField{T}) where T <: FloatOrComplex
    xp = parent(x.field)
    yp = parent(y.field)
    yp .+= s .* xp
    return y
end

function Krylov.kaxpby!(n::Integer, s::T, x::KrylovField{T}, t::T, y::KrylovField{T}) where T <: FloatOrComplex
    xp = parent(x.field)
    yp = parent(y.field)
    yp .= s .* xp .+ t .* yp
    return y
end

function Krylov.kscalcopy!(n::Integer, y::KrylovField{T}, s::T, x::KrylovField{T}) where T <: FloatOrComplex
    yp = parent(y.field)
    xp = parent(x.field)
    yp .= s .* xp
    return y
end

function Krylov.kdivcopy!(n::Integer, y::KrylovField{T}, x::KrylovField{T}, s::T) where T <: FloatOrComplex
    yp = parent(y.field)
    xp = parent(x.field)
    yp .= xp ./ s
    return y
end

Krylov.knorm(n::Integer, x::KrylovField{T}) where T <: FloatOrComplex = norm(x.field)
Krylov.kdot(n::Integer, x::KrylovField{T}, y::KrylovField{T}) where T <: FloatOrComplex = dot(x.field, y.field)
Krylov.kcopy!(n::Integer, y::KrylovField{T}, x::KrylovField{T}) where T <: FloatOrComplex = copyto!(y.field, x.field)
Krylov.kfill!(x::KrylovField{T}, val::T) where T <: FloatOrComplex = fill!(x.field, val)

## Structure representing linear operators so that we can define mul! on it
mutable struct KrylovOperator{T, F}
    type::Type{T}
    m::Int
    n::Int
    fun::F
    args::Tuple
end

## Structure representing preconditioners so that we can define mul! on it
mutable struct KrylovPreconditioner{T, P}
    type::Type{T}
    m::Int
    n::Int
    preconditioner::P
    args::Tuple
end

Base.size(A::KrylovOperator) = (A.m, A.n)
Base.eltype(A::KrylovOperator{T}) where T = T
LinearAlgebra.mul!(y::KrylovField, A::KrylovOperator, x::KrylovField) = A.fun(y.field, x.field, A.args...)

Base.size(P::KrylovPreconditioner) = (P.m, P.n)
Base.eltype(P::KrylovPreconditioner{T}) where T = T
LinearAlgebra.mul!(y::KrylovField, P::KrylovPreconditioner, x::KrylovField) = precondition!(y.field, P.preconditioner, x.field, P.args...)

## Solver using Krylov.jl
mutable struct KrylovSolver{A,G,L,S,P,T}
    architecture :: A
    grid :: G
    op :: L
    workspace :: S
    method :: Symbol
    preconditioner :: P
    abstol::T
    reltol::T
    maxiter::Int
    maxtime::Float64
end

architecture(solver::KrylovSolver) = solver.architecture
Base.summary(solver::KrylovSolver) = "KrylovSolver"

"""
    KrylovSolver(linear_operator;
                 template_field::AbstractField,
                 maxiter::Int = prod(size(template_field)),
                 maxtime::Real = Inf,
                 reltol::Real = sqrt(eps(eltype(template_field.grid))),
                 abstol::Real = zero(eltype(template_field.grid)),
                 preconditioner = nothing,
                 method::Symbol = :cg)

Construct a Krylov subspace solver for implicit linear systems defined by `linear_operator`,
using the structure of a reference field `template_field`.

# Arguments

- `linear_operator`: linear that defines the matrix-vector product `y = A * x`, where `x` has the same structure as `template_field`.
- `template_field::AbstractField`: A sample field used to infer the architecture, domain geometry, and data types. It is also used to allocate internal buffers and define the operator dimensions.
- `maxiter::Int`: Maximum number of Krylov iterations allowed.
- `maxtime::Real`: Maximum wall-clock time (in seconds) allowed for solving.
- `reltol::Real`: Relative tolerance on the residual norm for convergence.
- `abstol::Real`: Absolute tolerance on the residual norm for convergence.
- `preconditioner`: An optional preconditioner, passed as a callable or left as `nothing` for no preconditioning.
- `method::Symbol`: Krylov method to use, such as `:cg`, `:fom`, `:bicgstab`, `:gmres`.
"""
function KrylovSolver(linear_operator;
                      template_field::AbstractField,
                      maxiter = prod(size(template_field)),
                      maxtime = Inf,
                      reltol = sqrt(eps(eltype(template_field.grid))),
                      abstol = zero(eltype(template_field.grid)),
                      preconditioner = nothing,
                      method::Symbol = :cg)

    arch = architecture(template_field)
    grid = template_field.grid
    T = eltype(grid)

    # Linear operators
    m = n = length(template_field)
    op = KrylovOperator(T, m, n, linear_operator, ())
    P = preconditioner === nothing ? I : KrylovPreconditioner(T, m, n, preconditioner, ())

    kf = KrylovField(template_field)
    kc = Krylov.KrylovConstructor(kf)
    workspace = Krylov.krylov_workspace(Val(method), kc)

    return KrylovSolver(arch, grid, op, workspace, method, P, T(abstol), T(reltol), maxiter, maxtime)
end

function solve!(x, solver::KrylovSolver, b, args...; kwargs...)
    solver.op.args = args
    (solver.preconditioner === I) || (solver.preconditioner.args = args)
    if solver.method == :fom || solver.method == :gmres || solver.method == :fgmres || solver.method == :bicgstab
        # Use right preconditioning (keep invariant the residual norm)
        Krylov.krylov_solve!(solver.workspace, solver.op, KrylovField(b); N=solver.preconditioner,
                             atol=solver.abstol, rtol=solver.reltol, itmax=solver.maxiter,
                             timemax=solver.maxtime, kwargs...)
    else
        # Use left or centered preconditioning
        Krylov.krylov_solve!(solver.workspace, solver.op, KrylovField(b); M=solver.preconditioner,
                             atol=solver.abstol, rtol=solver.reltol, itmax=solver.maxiter,
                             timemax=solver.maxtime, kwargs...)
    end
    copyto!(x, solver.workspace.x.field)
    return x
end
