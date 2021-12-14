using SparseArrays, LinearAlgebra

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
#         mhat, Q, R =  find_mj_given_col(J, A, e, Q, R, 1)
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
#             mhat, Q, R = find_mj_given_col(J, A, e, Q, R, 0)
#             r = e - A[:, J] * mhat

#             @show j
#         end
#          mj      = spzeros(FT, n, 1)
#          mj[J]   = mhat
#          M[:, j] = mj
#     end
#     return M
# end

# function find_mj_given_col(col_el, A, ej, Q, R, restart)
    
#     A1     = A[:, col_el]
#     row_el = first.(Tuple.(findall(x -> x .!=0, A1)))
#     row_el = unique(row_el)

#     A1 = A1[row_el, :]
#     bj = ej[row_el, :]     
    
#     # if restart == 1
#         F = qr(A1)
#     # else
#     #     # update Q R factors
#     # end
#     mhat = minimize(F, bj, col_el)
#     return mhat, Q, R #
# end

# @inline minimize(F, bj, col_el) = (inv(Array(F.R)) * (F.Q' * bj)[1:length(col_el)])[sortperm(F.pcol)]

function spai_preconditioner(A::AbstractMatrix; ε)
    
    FT = eltype(A)
    n  = size(A, 1)
    M  = spzeros(FT, n, n)
    r  = ones(FT, n)
    Q = []
    R = []
    for j = 1 : n # this loop can be parallelized!
        @show j
        e = speyecolumn(FT, j, n)
        J = findall(x -> x .!=0, e)
        J̃ = []
        mhat =  find_mj_given_col(J, A, e)
        r = e - A[:, J] * mhat
        while norm(r) > ε
            L = first.(Tuple.(findall(x -> x .!=0, r)))
            J̃ = unique([L..., J...]) 
            # ρ = zeros(length(J̃))
            # for (t, k) in enumerate(J̃)
            #     ek   = speyecolumn(FT, k, n)
            #     ρ[t] = norm(r)^2 - norm(r' * A * ek)^2 / norm(A * ek)^2
            # end
            # J̃    = J̃[ ρ .< 0.1 ]
            J    = unique([J..., J̃...])
            mhat = find_mj_given_col(J, A, e)
            r = e - A[:, J] * mhat
        end
         mj      = spzeros(FT, n, 1)
         mj[J]   = mhat
         M[:, j] = mj
    end
    return M
end

function find_mj_given_col(col_el, A, ej)
    
    A1     = A[:, col_el]
    row_el = first.(Tuple.(findall(x -> x .!=0, A1)))
    row_el = unique(row_el)

    A1 = A1[row_el, :]
    bj = ej[row_el, :]     

    return A1 \ bj
end

@inline speye(FT, n) = spdiagm(0=>ones(FT, n))
@inline speyecolumn(FT, j, n) = [ FT(i == j) for i = 1 : n ]