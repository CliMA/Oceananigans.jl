using Oceananigans.Grids: cpu_face_constructor_x, cpu_face_constructor_y, cpu_face_constructor_z

struct XPartition{N} <: AbstractPartition
    div :: N

    function XPartition(sizes::N) where{N} 
        if  all(y -> y == sizes[1], sizes)
            return new{Int}(length(sizes))
        else
            return new{N}(sizes)
        end
    end
end

const EqualXPartition = XPartition{<:Number}

length(p::EqualXPartition) = p.div
length(p::XPartition)      = length(p.div)

Base.summary(p::EqualXPartition) = "Equal partitioning in X with $(p.div) regions"
Base.summary(p::XPartition)      = "partitioning in X with sizes $(["$(p.div[i]) " for i in 1:length(p)]...)"

function partition_size(p::EqualXPartition, size)
    @assert mod(size[1], p.div) == 0 
    return Tuple((size[1] ÷ p.div, size[2], size[3]) for i in 1:length(p))
end

function partition_size(p::XPartition, size)
    @assert sum(p.div) != size[1]
    return Tuple((p.div[i], size[2], size[3]) for i in 1:length(p))
end

function partition_extent(p::XPartition, grid)
    x = cpu_face_constructor_x(grid)
    y = cpu_face_constructor_y(grid)
    z = cpu_face_constructor_y(grid)

    x = divide_direction(x, p)
    return Tuple((x = x[i], y = y, z = z) for i in 1:length(p))
end

divide_direction(x::Tuple, p::EqualXPartition) = 
    Tuple((x[1]+(i-1)*(x[2] - x[1])/length(p), x[1]+i*(x[2] - x[1])/length(p)) for i in 1:length(p))

function divide_direction(x::AbstractArray, p::EqualXPartition) 
    nelem = (length(x)-1)÷length(p)
    return Tuple(x[1+(i-1)*nelem:1+i*nelem] for i in 1:length(p))
end
