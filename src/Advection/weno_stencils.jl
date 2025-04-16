
# WENO stencils
@inline z(T) = zero(T)

# WENO 3rd order limited to 1st order
@inline S₀₂(S::NTuple{N, T}, ::Val{1}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (S[2], z(T)), (S[3], z(T)))
@inline S₁₂(S::NTuple{N, T}, ::Val{1}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (z(T), z(T)), (z(T), z(T)))

@inline S₀₂(S::NTuple{N, T}, ::Val{2}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (S[2], S[3]), (S[3], S[2]))
@inline S₁₂(S::NTuple{N, T}, ::Val{2}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (S[1], S[2]), (S[4], S[3]))

# WENO 5th order limited to 3rd order, and 1st order
@inline S₀₃(S::NTuple{N, T}, ::Val{1}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (S[3], z(T), z(T)), (S[4], z(T), z(T)))
@inline S₁₃(S::NTuple{N, T}, ::Val{1}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (z(T), z(T), z(T)), (z(T), z(T), z(T)))
@inline S₂₃(S::NTuple{N, T}, ::Val{1}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (z(T), z(T), z(T)), (z(T), z(T), z(T)))

@inline S₀₃(S::NTuple{N, T}, ::Val{2}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (S[3], S[4], z(T)), (S[4], S[3], z(T)))
@inline S₁₃(S::NTuple{N, T}, ::Val{2}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (S[2], S[3], z(T)), (S[5], S[4], z(T)))
@inline S₂₃(S::NTuple{N, T}, ::Val{2}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (z(T), z(T), z(T)), (z(T), z(T), z(T)))

@inline S₀₃(S::NTuple{N, T}, ::Val{3}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (S[3], S[4], S[5]), (S[4], S[3], S[2]))
@inline S₁₃(S::NTuple{N, T}, ::Val{3}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (S[2], S[3], S[4]), (S[5], S[4], S[3]))
@inline S₂₃(S::NTuple{N, T}, ::Val{3}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (S[1], S[2], S[3]), (S[6], S[5], S[4]))

# WENO 7th order limited to 5th order, 3rd order, and 1st order
@inline S₀₄(S::NTuple{N, T}, ::Val{1}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (S[4], z(T), z(T), z(T)), (S[5], z(T), z(T), z(T)))
@inline S₁₄(S::NTuple{N, T}, ::Val{1}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (z(T), z(T), z(T), z(T)), (z(T), z(T), z(T), z(T)))
@inline S₂₄(S::NTuple{N, T}, ::Val{1}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (z(T), z(T), z(T), z(T)), (z(T), z(T), z(T), z(T)))
@inline S₃₄(S::NTuple{N, T}, ::Val{1}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (z(T), z(T), z(T), z(T)), (z(T), z(T), z(T), z(T)))

@inline S₀₄(S::NTuple{N, T}, ::Val{2}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (S[4], S[5], z(T), z(T)), (S[5], S[4], z(T), z(T)))
@inline S₁₄(S::NTuple{N, T}, ::Val{2}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (S[3], S[4], z(T), z(T)), (S[6], S[5], z(T), z(T)))
@inline S₂₄(S::NTuple{N, T}, ::Val{2}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (z(T), z(T), z(T), z(T)), (z(T), z(T), z(T), z(T)))
@inline S₃₄(S::NTuple{N, T}, ::Val{2}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (z(T), z(T), z(T), z(T)), (z(T), z(T), z(T), z(T)))

@inline S₀₄(S::NTuple{N, T}, ::Val{3}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (S[4], S[5], S[6], z(T)), (S[5], S[4], S[3], z(T)))
@inline S₁₄(S::NTuple{N, T}, ::Val{3}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (S[3], S[4], S[5], z(T)), (S[6], S[5], S[4], z(T)))
@inline S₂₄(S::NTuple{N, T}, ::Val{3}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (S[2], S[3], S[4], z(T)), (S[7], S[6], S[5], z(T)))
@inline S₃₄(S::NTuple{N, T}, ::Val{3}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (z(T), z(T), z(T), z(T)), (z(T), z(T), z(T), z(T)))

@inline S₀₄(S::NTuple{N, T}, ::Val{4}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (S[4], S[5], S[6], S[7]), (S[5], S[4], S[3], S[2]))
@inline S₁₄(S::NTuple{N, T}, ::Val{4}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (S[3], S[4], S[5], S[6]), (S[6], S[5], S[4], S[3]))
@inline S₂₄(S::NTuple{N, T}, ::Val{4}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (S[2], S[3], S[4], S[5]), (S[7], S[6], S[5], S[4]))
@inline S₃₄(S::NTuple{N, T}, ::Val{4}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (S[1], S[2], S[3], S[4]), (S[8], S[7], S[6], S[5]))

# WENO 9th order limited to 7th order, 5th order, 3rd order, and 1st order
@inline S₀₅(S::NTuple{N, T}, ::Val{1}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (S[5], z(T), z(T), z(T), z(T)), (S[6], z(T), z(T), z(T), z(T)))
@inline S₁₅(S::NTuple{N, T}, ::Val{1}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (z(T), z(T), z(T), z(T), z(T)), (z(T), z(T), z(T), z(T), z(T)))
@inline S₂₅(S::NTuple{N, T}, ::Val{1}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (z(T), z(T), z(T), z(T), z(T)), (z(T), z(T), z(T), z(T), z(T)))
@inline S₃₅(S::NTuple{N, T}, ::Val{1}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (z(T), z(T), z(T), z(T), z(T)), (z(T), z(T), z(T), z(T), z(T)))
@inline S₄₅(S::NTuple{N, T}, ::Val{1}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (z(T), z(T), z(T), z(T), z(T)), (z(T), z(T), z(T), z(T), z(T)))

@inline S₀₅(S::NTuple{N, T}, ::Val{2}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (S[5], S[6], z(T), z(T), z(T)), (S[6], S[5], z(T), z(T), z(T)))
@inline S₁₅(S::NTuple{N, T}, ::Val{2}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (S[4], S[5], z(T), z(T), z(T)), (S[7], S[6], z(T), z(T), z(T)))
@inline S₂₅(S::NTuple{N, T}, ::Val{2}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (z(T), z(T), z(T), z(T), z(T)), (z(T), z(T), z(T), z(T), z(T)))
@inline S₃₅(S::NTuple{N, T}, ::Val{2}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (z(T), z(T), z(T), z(T), z(T)), (z(T), z(T), z(T), z(T), z(T)))
@inline S₄₅(S::NTuple{N, T}, ::Val{2}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (z(T), z(T), z(T), z(T), z(T)), (z(T), z(T), z(T), z(T), z(T)))

@inline S₀₅(S::NTuple{N, T}, ::Val{3}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (S[5], S[6], S[7], z(T), z(T)), (S[6], S[5], S[4], z(T), z(T)))
@inline S₁₅(S::NTuple{N, T}, ::Val{3}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (S[4], S[5], S[6], z(T), z(T)), (S[7], S[6], S[5], z(T), z(T)))
@inline S₂₅(S::NTuple{N, T}, ::Val{3}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (S[3], S[4], S[5], z(T), z(T)), (S[8], S[7], S[6], z(T), z(T)))
@inline S₃₅(S::NTuple{N, T}, ::Val{3}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (z(T), z(T), z(T), z(T), z(T)), (z(T), z(T), z(T), z(T), z(T)))
@inline S₄₅(S::NTuple{N, T}, ::Val{3}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (z(T), z(T), z(T), z(T), z(T)), (z(T), z(T), z(T), z(T), z(T)))

@inline S₀₅(S::NTuple{N, T}, ::Val{4}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (S[5], S[6], S[7], S[8], z(T)), (S[6], S[5], S[4], S[3], z(T)))
@inline S₁₅(S::NTuple{N, T}, ::Val{4}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (S[4], S[5], S[6], S[7], z(T)), (S[7], S[6], S[5], S[4], z(T)))
@inline S₂₅(S::NTuple{N, T}, ::Val{4}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (S[3], S[4], S[5], S[6], z(T)), (S[8], S[7], S[6], S[5], z(T)))
@inline S₃₅(S::NTuple{N, T}, ::Val{4}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (S[2], S[3], S[4], S[5], z(T)), (S[9], S[8], S[7], S[6], z(T)))
@inline S₄₅(S::NTuple{N, T}, ::Val{4}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (z(T), z(T), z(T), z(T), z(T)), (z(T), z(T), z(T), z(T), z(T)))

@inline S₀₅(S::NTuple{N, T}, ::Val{5}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (S[5], S[6], S[7], S[8], S[9]), (S[6],  S[5], S[4], S[3], S[2]))
@inline S₁₅(S::NTuple{N, T}, ::Val{5}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (S[4], S[5], S[6], S[7], S[8]), (S[7],  S[6], S[5], S[4], S[3]))
@inline S₂₅(S::NTuple{N, T}, ::Val{5}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (S[3], S[4], S[5], S[6], S[7]), (S[8],  S[7], S[6], S[5], S[4]))
@inline S₃₅(S::NTuple{N, T}, ::Val{5}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (S[2], S[3], S[4], S[5], S[6]), (S[9],  S[8], S[7], S[6], S[5]))
@inline S₄₅(S::NTuple{N, T}, ::Val{5}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (S[1], S[2], S[3], S[4], S[5]), (S[10], S[9], S[8], S[7], S[6]))

# WENO 1th order limited to 9th order, 7th order, 5rd order, 3rd order, and 1st order
@inline S₀₆(S::NTuple{N, T}, ::Val{1}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (S[6], z(T), z(T), z(T), z(T),  z(T)),  (S[7],  z(T),  z(T),  z(T), z(T), z(T)))
@inline S₁₆(S::NTuple{N, T}, ::Val{1}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (z(T), z(T), z(T), z(T), z(T),  z(T)),  (z(T),  z(T),  z(T),  z(T), z(T), z(T)))
@inline S₂₆(S::NTuple{N, T}, ::Val{1}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (z(T), z(T), z(T), z(T), z(T),  z(T)),  (z(T),  z(T),  z(T),  z(T), z(T), z(T)))
@inline S₃₆(S::NTuple{N, T}, ::Val{1}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (z(T), z(T), z(T), z(T), z(T),  z(T)),  (z(T),  z(T),  z(T),  z(T), z(T), z(T)))
@inline S₄₆(S::NTuple{N, T}, ::Val{1}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (z(T), z(T), z(T), z(T), z(T),  z(T)),  (z(T),  z(T),  z(T),  z(T), z(T), z(T)))
@inline S₅₆(S::NTuple{N, T}, ::Val{1}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (z(T), z(T), z(T), z(T), z(T),  z(T)),  (z(T),  z(T),  z(T),  z(T), z(T), z(T)))

@inline S₀₆(S::NTuple{N, T}, ::Val{2}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (S[6], S[7], z(T), z(T), z(T),  z(T)),  (S[7],  S[6],  z(T),  z(T), z(T), z(T)))
@inline S₁₆(S::NTuple{N, T}, ::Val{2}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (S[5], S[6], z(T), z(T), z(T),  z(T)),  (S[8],  S[7],  z(T),  z(T), z(T), z(T)))
@inline S₂₆(S::NTuple{N, T}, ::Val{2}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (z(T), z(T), z(T), z(T), z(T),  z(T)),  (z(T),  z(T),  z(T),  z(T), z(T), z(T)))
@inline S₃₆(S::NTuple{N, T}, ::Val{2}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (z(T), z(T), z(T), z(T), z(T),  z(T)),  (z(T),  z(T),  z(T),  z(T), z(T), z(T)))
@inline S₄₆(S::NTuple{N, T}, ::Val{2}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (z(T), z(T), z(T), z(T), z(T),  z(T)),  (z(T),  z(T),  z(T),  z(T), z(T), z(T)))
@inline S₅₆(S::NTuple{N, T}, ::Val{2}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (z(T), z(T), z(T), z(T), z(T),  z(T)),  (z(T),  z(T),  z(T),  z(T), z(T), z(T)))

@inline S₀₆(S::NTuple{N, T}, ::Val{3}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (S[6], S[7], S[8], z(T), z(T),  z(T)),  (S[7],  S[6],  S[5],  z(T), z(T), z(T)))
@inline S₁₆(S::NTuple{N, T}, ::Val{3}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (S[5], S[6], S[7], z(T), z(T),  z(T)),  (S[8],  S[7],  S[6],  z(T), z(T), z(T)))
@inline S₂₆(S::NTuple{N, T}, ::Val{3}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (S[4], S[5], S[6], z(T), z(T),  z(T)),  (S[9],  S[8],  S[7],  z(T), z(T), z(T)))
@inline S₃₆(S::NTuple{N, T}, ::Val{3}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (z(T), z(T), z(T), z(T), z(T),  z(T)),  (z(T),  z(T),  z(T),  z(T), z(T), z(T)))
@inline S₄₆(S::NTuple{N, T}, ::Val{3}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (z(T), z(T), z(T), z(T), z(T),  z(T)),  (z(T),  z(T),  z(T),  z(T), z(T), z(T)))
@inline S₅₆(S::NTuple{N, T}, ::Val{3}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (z(T), z(T), z(T), z(T), z(T),  z(T)),  (z(T),  z(T),  z(T),  z(T), z(T), z(T)))

@inline S₁₆(S::NTuple{N, T}, ::Val{4}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (S[5], S[6], S[7], S[8], z(T),  z(T)),  (S[8],  S[7],  S[6],  S[5], z(T), z(T)))
@inline S₀₆(S::NTuple{N, T}, ::Val{4}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (S[6], S[7], S[8], S[9], z(T),  z(T)),  (S[7],  S[6],  S[5],  S[4], z(T), z(T)))
@inline S₂₆(S::NTuple{N, T}, ::Val{4}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (S[4], S[5], S[6], S[7], z(T),  z(T)),  (S[9],  S[8],  S[7],  S[6], z(T), z(T)))
@inline S₃₆(S::NTuple{N, T}, ::Val{4}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (S[3], S[4], S[5], S[6], z(T),  z(T)),  (S[10], S[9],  S[8],  S[7], z(T), z(T)))
@inline S₄₆(S::NTuple{N, T}, ::Val{4}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (z(T), z(T), z(T), z(T), z(T),  z(T)),  (z(T),  z(T),  z(T),  z(T), z(T), z(T)))
@inline S₅₆(S::NTuple{N, T}, ::Val{4}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (z(T), z(T), z(T), z(T), z(T),  z(T)),  (z(T),  z(T),  z(T),  z(T), z(T), z(T)))

@inline S₀₆(S::NTuple{N, T}, ::Val{5}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (S[6], S[7], S[8], S[9], S[10], z(T)),  (S[7],  S[6],  S[5],  S[4], S[3], z(T)))
@inline S₁₆(S::NTuple{N, T}, ::Val{5}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (S[5], S[6], S[7], S[8], S[9],  z(T)),  (S[8],  S[7],  S[6],  S[5], S[4], z(T)))
@inline S₂₆(S::NTuple{N, T}, ::Val{5}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (S[4], S[5], S[6], S[7], S[8],  z(T)),  (S[9],  S[8],  S[7],  S[6], S[5], z(T)))
@inline S₃₆(S::NTuple{N, T}, ::Val{5}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (S[3], S[4], S[5], S[6], S[7],  z(T)),  (S[10], S[9],  S[8],  S[7], S[6], z(T)))
@inline S₄₆(S::NTuple{N, T}, ::Val{5}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (S[2], S[3], S[4], S[5], S[6],  z(T)),  (S[11], S[10], S[9],  S[8], S[7], z(T)))
@inline S₅₆(S::NTuple{N, T}, ::Val{5}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (z(T), z(T), z(T), z(T), z(T),  z(T)),  (z(T),  z(T),  z(T),  z(T), z(T), z(T)))

@inline S₀₆(S::NTuple{N, T}, ::Val{6}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (S[6], S[7], S[8], S[9], S[10], S[11]), (S[7],  S[6],  S[5],  S[4], S[3], S[2]))
@inline S₁₆(S::NTuple{N, T}, ::Val{6}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (S[5], S[6], S[7], S[8], S[9],  S[10]), (S[8],  S[7],  S[6],  S[5], S[4], S[3]))
@inline S₂₆(S::NTuple{N, T}, ::Val{6}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (S[4], S[5], S[6], S[7], S[8],  S[9]),  (S[9],  S[8],  S[7],  S[6], S[5], S[4]))
@inline S₃₆(S::NTuple{N, T}, ::Val{6}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (S[3], S[4], S[5], S[6], S[7],  S[8]),  (S[10], S[9],  S[8],  S[7], S[6], S[5]))
@inline S₄₆(S::NTuple{N, T}, ::Val{6}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (S[2], S[3], S[4], S[5], S[6],  S[7]),  (S[11], S[10], S[9],  S[8], S[7], S[6]))
@inline S₅₆(S::NTuple{N, T}, ::Val{6}, bias) where {N, T} = @inbounds ifelse(bias isa LeftBias, (S[1], S[2], S[3], S[4], S[5],  S[6]),  (S[12], S[11], S[10], S[9], S[8], S[7]))
