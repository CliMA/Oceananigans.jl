const AF = AbstractField

# Two-dimensional fields
@inline Base.getindex( r::AF{Nothing, Y, Z},    i, j, k) where {Y, Z} = @inbounds r.data[1, j, k]
@inline Base.setindex!(r::AF{Nothing, Y, Z}, d, i, j, k) where {Y, Z} = @inbounds r.data[1, j, k] = d
@inline Base.getindex( r::AF{X, Nothing, Z},    i, j, k) where {X, Z} = @inbounds r.data[i, 1, k]
@inline Base.setindex!(r::AF{X, Nothing, Z}, d, i, j, k) where {X, Z} = @inbounds r.data[i, 1, k] = d
@inline Base.getindex( r::AF{X, Y, Nothing},    i, j, k) where {X, Y} = @inbounds r.data[i, j, 1]
@inline Base.setindex!(r::AF{X, Y, Nothing}, d, i, j, k) where {X, Y} = @inbounds r.data[i, j, 1] = d

@inline Base.getindex( r::AF{Nothing, Y, Z},    j, k) where {Y, Z} = @inbounds r.data[1, j, k]
@inline Base.setindex!(r::AF{Nothing, Y, Z}, d, j, k) where {Y, Z} = @inbounds r.data[1, j, k] = d
@inline Base.getindex( r::AF{X, Nothing, Z},    i, k) where {X, Z} = @inbounds r.data[i, 1, k]
@inline Base.setindex!(r::AF{X, Nothing, Z}, d, i, k) where {X, Z} = @inbounds r.data[i, 1, k] = d
@inline Base.getindex( r::AF{X, Y, Nothing},    i, j) where {X, Y} = @inbounds r.data[i, j, 1]
@inline Base.setindex!(r::AF{X, Y, Nothing}, d, i, j) where {X, Y} = @inbounds r.data[i, j, 1] = d

# One-dimensional fields
@inline Base.getindex( r::AF{X, Nothing, Nothing},    i, j, k) where X = @inbounds r.data[i, 1, 1]
@inline Base.setindex!(r::AF{X, Nothing, Nothing}, d, i, j, k) where X = @inbounds r.data[i, 1, 1] = d
@inline Base.getindex( r::AF{Nothing, Y, Nothing},    i, j, k) where Y = @inbounds r.data[1, j, 1]
@inline Base.setindex!(r::AF{Nothing, Y, Nothing}, d, i, j, k) where Y = @inbounds r.data[1, j, 1] = d
@inline Base.getindex( r::AF{Nothing, Nothing, Z},    i, j, k) where Z = @inbounds r.data[1, 1, k]
@inline Base.setindex!(r::AF{Nothing, Nothing, Z}, d, i, j, k) where Z = @inbounds r.data[1, 1, k] = d

@inline Base.getindex( r::AF{X, Nothing, Nothing},    i) where X = @inbounds r.data[i, 1, 1]
@inline Base.setindex!(r::AF{X, Nothing, Nothing}, d, i) where X = @inbounds r.data[i, 1, 1] = d
@inline Base.getindex( r::AF{Nothing, Y, Nothing},    j) where Y = @inbounds r.data[1, j, 1]
@inline Base.setindex!(r::AF{Nothing, Y, Nothing}, d, j) where Y = @inbounds r.data[1, j, 1] = d
@inline Base.getindex( r::AF{Nothing, Nothing, Z},    k) where Z = @inbounds r.data[1, 1, k]
@inline Base.setindex!(r::AF{Nothing, Nothing, Z}, d, k) where Z = @inbounds r.data[1, 1, k] = d

# Zero-dimensional fields
@inline Base.getindex( r::AF{Nothing, Nothing, Nothing},    i, j, k) = @inbounds r.data[1, 1, 1]
@inline Base.setindex!(r::AF{Nothing, Nothing, Nothing}, d, i, j, k) = @inbounds r.data[1, 1, 1] = d

@inline Base.getindex( r::AF{Nothing, Nothing, Nothing},  ) = @inbounds r.data[1, 1, 1]
@inline Base.setindex!(r::AF{Nothing, Nothing, Nothing}, d) = @inbounds r.data[1, 1, 1] = d


