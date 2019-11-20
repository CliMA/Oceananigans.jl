struct BatchedTridiagonalSolver{A, B, C, D}
    a :: A
    b :: B
    c :: C
    d :: D
end

@inline get_coefficient(a::AbstractArray{T, 1}, i, j, k) where {T} = @inbounds a[k]
@inline get_coefficient(a::AbstractArray{T, 3}, i, j, k) where {T} = @inbounds a[i, j, k]
@inline get_coefficient(a::Function, i, j, k) = a(i, j, k)


