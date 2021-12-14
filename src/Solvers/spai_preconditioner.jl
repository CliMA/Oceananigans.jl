function spai_preconditioner(A::AbstractMatrix; ε = 0.2)
    
    n  = size(A, 1)
    M  = spzeros(n, n)
    r  = ones(n)
    for j = 1 : n # this loop can be parallelized!
        @show j
        e = speyecolumn(j, n)
        J = findall(x -> x .!=0, e)
        J̃ = []
        mhat =  find_mj_given_col(J, A, e)
        r = e - A[:, J] * mhat
        while norm(r) > ε
            L = first.(Tuple.(findall(x -> x .!=0, r)))
            J̃ = unique([L..., J...]) 
            # ρ = zeros(length(J̃))
            # for (t, k) in enumerate(J̃)
            #     ek   = speyecolumn(k, n)
            #     ρ[t] = norm(r)^2 - norm(r' * A * ek)^2 / norm(A * ek)^2
            # end
            J    = unique([J..., J̃...])
            mhat = find_mj_given_col(J, A, e)
            r = e - A[:, J] * mhat
        end
         mj      = spzeros(n, 1)
         mj[J]   = mhat
         M[j, :] = mj
    end
    return M
end


function find_mj_given_col(col_el, A, ej)
    
    A1     = A[:, col_el]
    row_el = first.(Tuple.(findall(x -> x .!=0, A1)))
    row_el = unique(row_el)
    A1 = A1[row_el, :]
    bj = ej[row_el, :]
    
    @show size(A1)
    return A1 \ bj
end

function calc_row_and_matr_given_col(A, col_el)
    A1 = A[:, col_el]
    row_el  = first.(Tuple.(findall(x -> x .!=0, A1)))
    return A1, unique(row_el)
end

function sappinv(A, Apriori)
    
    n  = size(A, 1)
    Id = speye(n)
    M  = spzeros(n,n)
    
    for j = 1 : n #use parfor
        sj = Id[:, j]
        
        # find the non-zero rows of a priori pattern
        col_el  = findall(x -> x .!= 0, Apriori[j, :])
        A1      = A[:, col_el]
        
        rows_el = first.(Tuple.(findall(x -> x .!=0, A1)))

        rows_el = unique(rows_el)
        A1 = A1[rows_el, :]
        sj = Array(sj[rows_el, :])
     
        # the QR implementation
        bj = A1 \ sj
                
        mj = spzeros(n, 1)
        mj[col_el] = bj
        M[:, j]    = mj
    end

    return M
end

@inline speye(n) = spdiagm(0=>ones(n))
@inline speyecolumn(j, n) = [ Int(i == j) for i = 1 : n]