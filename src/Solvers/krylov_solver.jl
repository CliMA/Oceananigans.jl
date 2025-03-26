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

Krylov.knorm(n::Integer, x::KrylovField{T}) where T <: FloatOrComplex = norm(x.field)
Krylov.kdot(n::Integer, x::KrylovField{T}, y::KrylovField{T}) where T <: FloatOrComplex = dot(x.field, y.field)
Krylov.kcopy!(n::Integer, y::KrylovField{T}, x::KrylovField{T}) where T <: FloatOrComplex = copyto!(y.field, x.field)
Krylov.kfill!(x::KrylovField{T}, val::T) where T <: FloatOrComplex = fill!(x.field, val)

# Only needed by the Krylov solver MINRES-QLP.
# We can implement a kernel if we need it.
#
# function Krylov.kref!(n::Integer, x::KrylovField{T}, y::KrylovField{T}, c::T, s::T) where T <: FloatOrComplex
#     mx, nx, kx = size(x.field)
#     _x = x.field
#     _y = y.field
#     for i = 1:mx
#         for j = 1:nx
#             for k = 1:kx
#                 x_ijk = _x[i,j,k]
#                 y_ijk = _y[i,j,k]
#                 _x[i,j,k] = c       * x_ijk + s * y_ijk
#                 _y[i,j,k] = conj(s) * x_ijk - c * y_ijk
#             end
#         end
#     end
#     return x, y
# end

## Structure representing linear operators so that we can define mul! on it
struct KrylovOperator{T, F}
    type::Type{T}
    m::Int
    n::Int
    fun::F
    args::Tuple
end

## Structure representing preconditioners so that we can define mul! on it
struct KrylovPreconditioner{T, P}
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
    krylov_solver :: Symbol
    preconditioner :: P
    abstol::T
    reltol::T
    maxiter::Int
    maxtime::Float64
end

architecture(solver::KrylovSolver) = solver.architecture
Base.summary(solver::KrylovSolver) = "KrylovSolver"

function KrylovSolver(linear_operator;
                      template_field::AbstractField,
                      maxiter = prod(size(template_field)),
                      maxtime = Inf,
                      reltol = sqrt(eps(eltype(template_field.grid))),
                      abstol = zero(eltype(template_field.grid)),
                      preconditioner = nothing,
                      krylov_solver::Symbol = :cg)

    arch = architecture(template_field)
    grid = template_field.grid
    T = eltype(grid)

    # Linear operators
    m = n = length(template_field)
    op = KrylovOperator(T, m, n, linear_operator, ())
    P = preconditioner === nothing ? I : KrylovPreconditioner(T, m, n, preconditioner, ())

    kf = KrylovField(template_field)
    kc = Krylov.KrylovConstructor(kf)
    workspace = Krylov.eval(Krylov.KRYLOV_SOLVERS[krylov_solver])(kc)

    return KrylovSolver(arch, grid, op, workspace, krylov_solver, P, T(abstol), T(reltol), maxiter, maxtime)
end

function solve!(x, solver::KrylovSolver, b, args...; kwargs...)
    solver.op.args = args
    solver.preconditioner.args = args
    Krylov.solve!(solver.workspace, solver.op, KrylovField(b); M=solver.preconditioner,
                  atol=solver.abstol, rtol=solver.reltol, itmax=solver.maxiter, timemax=solver.maxtime, kwargs...)
    copyto!(x, solver.workspace.x.field)
    return x
end
