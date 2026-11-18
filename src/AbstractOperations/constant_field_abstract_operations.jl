#####
##### AbstractOperations with ZeroField and ConstantField
#####
##### Here we define a small number of operations with ZeroField
##### and ConstantField. Arbitrary AbstractOperations may not work.
#####

using Oceananigans.Fields: ZeroField, ConstantField

# Binary operations
Base.:(==)(::ZeroField, ::ZeroField) = true

Base.:(==)(zf::ZeroField, cf::ConstantField) = 0 == cf.constant
Base.:(==)(cf::ConstantField, zf::ZeroField) = 0 == cf.constant
Base.:(==)(c1::ConstantField, c2::ConstantField) = c1.constant == c2.constant

Base.:+(a::ZeroField, b::AbstractField) = b
Base.:+(a::AbstractField, b::ZeroField) = a
Base.:+(a::ZeroField, b::Number) = ConstantField(b)
Base.:+(a::Number, b::ZeroField) = ConstantField(a)

Base.:-(a::ZeroField, b::AbstractField) = -b
Base.:-(a::AbstractField, b::ZeroField) = a
Base.:-(a::ZeroField, b::Number) = ConstantField(-b)
Base.:-(a::Number, b::ZeroField) = ConstantField(a)

Base.:*(a::ZeroField, b::AbstractField) = a
Base.:*(a::AbstractField, b::ZeroField) = b
Base.:*(a::ZeroField, b::Number) = a
Base.:*(a::Number, b::ZeroField) = b

Base.:/(a::ZeroField, b::AbstractField) = a
Base.:/(a::AbstractField, b::ZeroField) = ConstantField(convert(eltype(a), Inf))
Base.:/(a::ZeroField, b::Number) = a
Base.:/(a::Number, b::ZeroField) = ConstantField(a / convert(eltype(a), 0))

# ZeroField as the constant zero next to a ConstantField
Base.:+(a::ZeroField, b::ConstantField) = b
Base.:+(a::ConstantField, b::ZeroField) = a
Base.:-(a::ZeroField, b::ConstantField) = ConstantField(-b.constant)
Base.:-(a::ConstantField, b::ZeroField) = a
Base.:*(a::ZeroField, b::ConstantField) = a
Base.:*(a::ConstantField, b::ZeroField) = b
Base.:/(a::ZeroField, b::ConstantField) = a
Base.:/(a::ConstantField, b::ZeroField) = ConstantField(a.constant / zero(eltype(b)))

function Base.:/(z1::ZeroField{T1, N1}, z2::ZeroField{T2, N2}) where {T1, T2, N1, N2}
    T = Base.promote_type(T1, T2)
    return ConstantField{max(N1, N2)}(zero(T) / zero(T))
end

# for two ZeroField
for op in (:-, :+, :*)
    @eval begin
        function Base.$op(z1::ZeroField{T1, N1}, z2::ZeroField{T2, N2}) where {T1, T2, N1, N2}
            T = Base.promote_type(T1, T2)
            N = max(N1, N2)
            return ZeroField{T, N}()
        end
    end
end

# Unary operations
Base.:-(a::ZeroField) = a
