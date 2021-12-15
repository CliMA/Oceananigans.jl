using SparseArrays, LinearAlgebra, Statistics

mutable struct SpaiIterator{Tv, Ti, Vi}
     mhat :: AbstractVector{Tv}
        e :: SparseVector{Tv}
        r :: SparseVector{Tv}
        J :: AbstractVector{Ti}
        I :: AbstractVector{Ti}
        J̃ :: AbstractVector{Ti}
        Ĩ :: AbstractVector{Ti}
        Q :: SparseMatrixCSC{Tv, Vi}
        R :: SparseMatrixCSC{Tv, Vi}
end

"""
The SPAI preconditioner calculates a SParse Approximate Inverse M ≈ A⁻¹
to be used as a preconditioner

The algorithm implemeted here to calculates M following the specifications found in

Grote M. J. & Huckle T, "Parallel Preconditioning with sparse approximate inverses" 


spai_preconditioner(A::AbstractMatrix; ε)

returns a SparseInversePreconditioner(M) where M ≈ A⁻¹
"""


# function spai_preconditioner(A::AbstractMatrix; ε)
    
#     FT = eltype(A)
#     n  = size(A, 1)
#     M  = spzeros(FT, n, n)
#     r  = ones(FT, n)
#     Q = []
#     R = []
#     for j = 1 : n # this loop can be parallelized! (not on GPU unfortunately...)
#         @show j
#         e = speyecolumn(FT, j, n)
#         J = findall(x -> x .!=0, e)
#         J̃ = []
#         mhat, Q, R, I =  initialize_m_Q_R(J, A, e)
#         r = e - A[:, J] * mhat
#         while norm(r) > ε 
#             L = first.(Tuple.(findall(x -> x .!=0, r)))
#             J̃ = unique([L..., J...]) 
#             mhat, Q, R = find_mj_given_col(J̃, J, A, e, Q, R, I)
#             J    = unique([J..., J̃...])
#             r = e - A[:, J] * mhat
#         end
#          mj      = spzeros(FT, n, 1)
#          mj[J]   = mhat
#          M[:, j] = mj
#     end
#     return M
# end

# function find_mj_given_col(col_el, col_el_old, A, ej, Q, R)
    
#     A1     = A[:, col_el]
#     row_el = first.(Tuple.(findall(x -> x .!=0, A1)))
#     row_el = unique(row_el)
    
#     A2         = A[:, col_el_old]
#     row_el_old = first.(Tuple.(findall(x -> x .!=0, A2)))
#     row_el_old = unique(row_el_old)

#     A1 = A1[row_el, :]
#     bj = ej[row_el, :]     
    
#     # if restart == 1
#         F = qr(A1, ordering = false)
#         Q, R = F.Q, UpperTriangular(F.R)
#     # else

#     #     # update Q R factors
#     # end

#     mhat = minimize(Q, R, bj, col_el)
#     return mhat, Q, R 
# end

# function initilize_m_Q_R(col_el, A, ej)
    
#     A1     = A[:, col_el]
#     row_el = first.(Tuple.(findall(x -> x .!=0, A1)))
#     row_el = unique(row_el)

#     A1 = A1[row_el, :]
#     bj = ej[row_el, :]     
    
#     F = qr(A1, ordering = false)
#     Q, R = F.Q, UpperTriangular(F.R)
#     mhat = minimize(Q, R, bj, col_el)
#     return mhat, Q, R, row_el
# end


# function spai_preconditioner(A::AbstractMatrix; ε)
    
#     FT = eltype(A)
#     n  = size(A, 1)
#     M  = spzeros(FT, n, n)
#     r  = ones(FT, n)
#     Q = []
#     R = []
#     for j = 1 : n # this loop can be parallelized!
#         @show j
#         e = speyecolumn(FT, j, n)
#         J = findall(x -> x .!=0, e)
#         J̃ = []
#         mhat =  find_mj_given_col(J, A, e)
#         r = e - A[:, J] * mhat
#         while norm(r) > ε
#             L = first.(Tuple.(findall(x -> x .!=0, r)))
#             J̃ = unique([L..., J...]) 
#             # ρ = zeros(length(J̃))
#             # for (t, k) in enumerate(J̃)
#             #     ek   = speyecolumn(FT, k, n)
#             #     ρ[t] = norm(r)^2 - norm(r' * A * ek)^2 / norm(A * ek)^2
#             # end
#             # J̃    = J̃[ ρ .< 0.1 ]
#             J    = unique([J..., J̃...])
#             mhat = find_mj_given_col(J, A, e)
#             r = e - A[:, J] * mhat
#         end
#          mj      = spzeros(FT, n, 1)
#          mj[J]   = mhat
#          M[:, j] = mj
#     end
#     return M
# end

# function find_mj_given_col(col_el, A, ej)
    
#     A1     = A[:, col_el]
#     row_el = first.(Tuple.(findall(x -> x .!=0, A1)))
#     row_el = unique(row_el)

#     A1 = A1[row_el, :]
#     bj = ej[row_el, :]     

#     return A1 \ bj
# end



### This works!!

# function set_j_column(A, j, ε, n, FT)
#     # this loop can be parallelized!
#     e = speyecolumn(FT, j, n)
#     r = zeros(FT, n)
#     J = findall(x -> x .!=0, e)
#     iterator = SpaiIterator(e, e, r, J, J, J, J, [], [])
#     find_mhat_given_col!(iterator, A)
#     calc_residuals!(iterator, A)
#     while norm(iterator.r) > ε 
#         L = first.(Tuple.(findall(x -> x .!=0, iterator.r)))
#         iterator.J̃ = unique([L..., iterator.J...]) 
#         iterator.J = unique([iterator.J..., iterator.J̃...])
#         find_mhat_given_col!(iterator, A)
#         calc_residuals!(iterator, A)
#     end
#     return iterator.mhat, iterator.J
# end    

# function update_mhat_given_col!(iterator, A, FT)
#     @inbounds begin
#         A1  = A[:, iterator.J̃]
#         A1I = A1[iterator.I, :]
        
#         Jₜₒₜ = [iterator.J..., iterator.J̃...]
#         row_el     = first.(Tuple.(findall(x -> x .!=0, A[:, Jₜₒₜ])))
#         iterator.Ĩ = setdiff(unique(row_el), iterator.I)

#         A1Ĩ = A1[iterator.Ĩ, :]
    
#         n₁ = length(iterator.I)
#         n₂ = length(iterator.J)
#         ñ₁ = length(iterator.Ĩ)
#         ñ₂ = length(iterator.J̃)

#         B1 = spzeros(n₁ + ñ₁, n₁ + ñ₁)
#         mul!(B1, iterator.Q[:,1:n₂]', A1I)
#         B2 = iterator.Q[:,n₂+1:end]' * A1I
#         B2 = sparse(vcat(B2, A1Ĩ))

#         # update_QR_decomposition!(iterator.Q, iterator.R, B1, B2, n₁, n₂, ñ₁, ñ₂)
#         F = qr(B2, ordering = false)

#         Iₙ₁ = speye(FT, ñ₁)
#         Iₙ₂ = speye(FT, n₂)
#         hm = spzeros(n₁, ñ₁)
#         iterator.Q = vcat(hcat(iterator.Q, hm), hcat(hm', Iₙ₁))
#         hm = spzeros(ñ₁ + n₁ - n₂, n₂)
#         iterator.Q = iterator.Q * vcat(hcat(Iₙ₂, hm'), hcat(hm, sparse(F.Q)))
        
#         hm = spzeros(ñ₂, n₂)
#         iterator.R = UpperTriangular(vcat(hcat(iterator.R, B1), hcat(hm, F.R)))
        
#         iterator.J = Jₜₒₜ
#         iterator.I = [iterator.I..., iterator.Ĩ...]

#         bj = zeros(length(iterator.I))
#         copyto!(bj, iterator.e[iterator.I])
        
#         minimize!(iterator, bj)
#     end
# end

# function find_mhat_given_col!(iterator, A)
    
#     A1         = A[:, iterator.J]
#     row_el     = first.(Tuple.(findall(x -> x .!=0, A1)))
#     iterator.I = unique(row_el)

#     A1 = A1[iterator.I, :]
#     bj = iterator.e[iterator.I, :]
    
#     F = qr(A1, ordering = false)
#     iterator.Q = sparse(F.Q)
#     iterator.R = UpperTriangular(F.R)

#     minimize!(iterator, bj)
# end


## This also!!

function spai_preconditioner(A::AbstractMatrix; ε)
    
    FT = eltype(A)
    n  = size(A, 1)
    M  = spzeros(FT, n, n)
    for j = 1 : n 
        @show j
        mhat, J = set_j_column(A, j, ε, n, FT)
        mj      = spzeros(FT, n, 1)
        mj[J]   = mhat
        M[:, j] = mj
    end
    return M
end

function set_j_column(A, j, ε, n, FT)
    # this loop can be parallelized!
    @inbounds begin
        e = speyecolumn(FT, j, n)
        r = spzeros(FT, n)
        J = A.rowval[A.colptr[j]:A.colptr[j+1]-1]
        Q = spzeros(FT, 1, 1)
        iterator = SpaiIterator(e, e, r, J, J, J, J, Q, Q)
        find_mhat_given_col!(iterator, A, n)
        calc_residuals!(iterator, A)
        while norm(iterator.r) > ε 
            iterator.J̃ = setdiff(iterator.r.nzind, iterator.J)
            update_mhat_given_col!(iterator, A, FT)
            calc_residuals!(iterator, A)
        end
    end
    return iterator.mhat, iterator.J
end    

function update_mhat_given_col!(iterator, A, FT)
    @inbounds begin
        A1  = A[:, iterator.J̃]
        A1I = A1[iterator.I, :]
        
        Jₜₒₜ = [iterator.J..., iterator.J̃...]
        row_el     = first.(Tuple.(findall(x -> x .!=0, A[:, Jₜₒₜ])))
        iterator.Ĩ = setdiff(unique(row_el), iterator.I)

        A1Ĩ = A1[iterator.Ĩ, :]
    
        n₁ = length(iterator.I)
        n₂ = length(iterator.J)
        ñ₁ = length(iterator.Ĩ)
        ñ₂ = length(iterator.J̃)

        B1 = spzeros(n₂, ñ₂)
        mul!(B1, iterator.Q[:,1:n₂]', A1I)
        B2 = iterator.Q[:,n₂+1:end]' * A1I
        B2 = sparse(vcat(B2, A1Ĩ))

        # update_QR_decomposition!(iterator.Q, iterator.R, B1, B2, n₁, n₂, ñ₁, ñ₂)
        F = qr(B2, ordering = false)

        Iₙ₁ = speye(FT, ñ₁)
        Iₙ₂ = speye(FT, n₂)
        hm = spzeros(n₁, ñ₁)
        iterator.Q = vcat(hcat(iterator.Q, hm), hcat(hm', Iₙ₁))
        hm = spzeros(ñ₁ + n₁ - n₂, n₂)
        iterator.Q = iterator.Q * vcat(hcat(Iₙ₂, hm'), hcat(hm, sparse(F.Q)))
        
        hm = spzeros(ñ₂, n₂)
        iterator.R = vcat(hcat(iterator.R, B1), hcat(hm, F.R))
        
        push!(iterator.J, iterator.J̃...)
        push!(iterator.I, iterator.Ĩ...)

        bj = zeros(length(iterator.I))
        copyto!(bj, iterator.e[iterator.I])
        
        minimize!(iterator, bj)
    end
end

function find_mhat_given_col!(iterator, A, n)
    
    A1 = spzeros(n, length(iterator.J))
    copyto!(A1, A[:, iterator.J])
    
    iterator.I = unique(A1.rowval)

    bj = zeros(length(iterator.I))
    copyto!(bj, iterator.e[iterator.I])
    
    F = qr(A1[iterator.I, :], ordering = false)
    iterator.Q = sparse(F.Q)
    iterator.R = sparse(F.R)
    
    minimize!(iterator, bj)
end

@inline calc_residuals!(i::SpaiIterator, A) = i.r = i.e - A[:, i.J] * i.mhat
@inline minimize!(i::SpaiIterator, bj)      = i.mhat = (i.R \ (i.Q' * bj)[1:length(i.J)])
@inline speye(FT, n) = spdiagm(0=>ones(FT, n))

@inline function speyecolumn(FT, j, n) 
    e    = spzeros(FT, n)
    e[j] = FT(1)
    return e
end
