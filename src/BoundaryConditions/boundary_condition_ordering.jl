
retrieve_bc(bc) = bc

# Returns the boundary conditions a specific side for `FieldBoundaryConditions` inputs and
# a tuple of boundary conditions for `NTuple{N, <:FieldBoundaryConditions}` inputs
for dir in (:west, :east, :south, :north, :bottom, :top)
    extract_side_bc = Symbol(:extract_, dir, :_bc)
    @eval begin
        @inline $extract_side_bc(bc) = retrieve_bc(bc.$dir)
        @inline $extract_side_bc(bc::Tuple) = map($extract_side_bc, bc)
    end
end
#
@inline extract_bc(bc, ::Val{:west})   = tuple(extract_west_bc(bc))
@inline extract_bc(bc, ::Val{:east})   = tuple(extract_east_bc(bc))
@inline extract_bc(bc, ::Val{:south})  = tuple(extract_south_bc(bc))
@inline extract_bc(bc, ::Val{:north})  = tuple(extract_north_bc(bc))
@inline extract_bc(bc, ::Val{:bottom}) = tuple(extract_bottom_bc(bc))
@inline extract_bc(bc, ::Val{:top})    = tuple(extract_top_bc(bc))

@inline extract_bc(bc, ::Val{:west_and_east})   = (extract_west_bc(bc), extract_east_bc(bc))
@inline extract_bc(bc, ::Val{:south_and_north}) = (extract_south_bc(bc), extract_north_bc(bc))
@inline extract_bc(bc, ::Val{:bottom_and_top})  = (extract_bottom_bc(bc), extract_top_bc(bc))

# In case of a DistributedCommunication paired with a
# Flux, Value or Gradient boundary condition, we split the direction in two single-sided
# fill_halo! events (see issue #3342)
# `permute_boundary_conditions` returns a 2-tuple containing the ordered operations to execute in
# position [1] and the associated boundary conditions in position [2]
function permute_boundary_conditions(boundary_conditions)

    split_x_halo_filling = split_halo_filling(extract_west_bc(boundary_conditions),  extract_east_bc(boundary_conditions))
    split_y_halo_filling = split_halo_filling(extract_south_bc(boundary_conditions), extract_north_bc(boundary_conditions))

    west_bc  = extract_west_bc(boundary_conditions)
    east_bc  = extract_east_bc(boundary_conditions)
    south_bc = extract_south_bc(boundary_conditions)
    north_bc = extract_north_bc(boundary_conditions)

    if split_x_halo_filling
        if split_y_halo_filling
            fill_halos! = [fill_west_halo!, fill_east_halo!, fill_south_halo!, fill_north_halo!, fill_bottom_and_top_halo!]
            sides       = [:west, :east, :south, :north, :bottom_and_top]
            bcs_array   = [west_bc, east_bc, south_bc, north_bc, extract_bottom_bc(boundary_conditions)]
        else
            fill_halos! = [fill_west_halo!, fill_east_halo!, fill_south_and_north_halo!, fill_bottom_and_top_halo!]
            sides       = [:west, :east, :south_and_north, :bottom_and_top]
            bcs_array   = [west_bc, east_bc, south_bc, extract_bottom_bc(boundary_conditions)]
        end
    else
        if split_y_halo_filling
            fill_halos! = [fill_west_and_east_halo!, fill_south_halo!, fill_north_halo!, fill_bottom_and_top_halo!]
            sides       = [:west_and_east, :south, :north, :bottom_and_top]
            bcs_array   = [west_bc, south_bc, north_bc, extract_bottom_bc(boundary_conditions)]
        else
            fill_halos! = [fill_west_and_east_halo!, fill_south_and_north_halo!, fill_bottom_and_top_halo!]
            sides       = [:west_and_east, :south_and_north, :bottom_and_top]
            bcs_array   = [west_bc, south_bc, extract_bottom_bc(boundary_conditions)]
        end
    end

    perm = sortperm(bcs_array, lt=fill_first)
    fill_halos! = fill_halos![perm]
    sides = sides[perm]

    boundary_conditions = Tuple(extract_bc(boundary_conditions, Val(side)) for side in sides)

    return fill_halos!, boundary_conditions
end

# Split direction in two distinct fill_halo! events in case of a communication boundary condition
# (distributed DCBC), paired with a Flux, Value or Gradient boundary condition
split_halo_filling(bcs1, bcs2)     = false
split_halo_filling(::DCBC, ::DCBC) = false
split_halo_filling(bcs1, ::DCBC)   = true
split_halo_filling(::DCBC, bcs2)   = true

# TODO: support heterogeneous distributed-shared communication
# split_halo_filling(::MCBC, ::DCBC) = false
# split_halo_filling(::DCBC, ::MCBC) = false
# split_halo_filling(::MCBC, ::MCBC) = false
# split_halo_filling(bcs1, ::MCBC)   = true
# split_halo_filling(::MCBC, bcs2)   = true

#####
##### Halo filling order
#####

const PBCT  = Union{PBC,  Tuple{Vararg{PBC}}}
const MCBCT = Union{MCBC, Tuple{Vararg{MCBC}}}
const DCBCT = Union{DCBC, Tuple{Vararg{DCBC}}}

# Distributed halos have to be filled last to allow the
# possibility of asynchronous communication:
# If other halos are filled after we initiate the distributed communication,
# (but before communication is completed) the halos will be overwritten.
# For this reason we always want to perform local halo filling first and then
# initiate communication

# Periodic is handled after Flux, Value, Gradient because
# Periodic fills also corners while Flux, Value, Gradient do not
# TODO: remove this ordering requirement (see issue https://github.com/CliMA/Oceananigans.jl/issues/3342)

# Order of halo filling
# 1) Flux, Value, Gradient (TODO: remove these BC and apply them as fluxes)
# 2) Periodic (PBCT)
# 3) Shared Communication (MCBCT)
# 4) Distributed Communication (DCBCT)

# We define "greater than" `>` and "lower than", for boundary conditions
# following the rules outlined in `fill_first`
# i.e. if `bc1 > bc2` then `bc2` precedes `bc1` in filling order
@inline Base.isless(bc1::BoundaryCondition, bc2::BoundaryCondition) = fill_first(bc1, bc2)

# fallback for `Nothing` BC.
@inline Base.isless(::Nothing,           ::Nothing) = true
@inline Base.isless(::BoundaryCondition, ::Nothing) = false
@inline Base.isless(::Nothing, ::BoundaryCondition) = true
@inline Base.isless(::BoundaryCondition, ::Missing) = false
@inline Base.isless(::Missing, ::BoundaryCondition) = true

fill_first(bc1::DCBCT, bc2)        = false
fill_first(bc1::PBCT,  bc2::DCBCT) = true
fill_first(bc1::DCBCT, bc2::PBCT)  = false
fill_first(bc1::MCBCT, bc2::DCBCT) = true
fill_first(bc1::DCBCT, bc2::MCBCT) = false
fill_first(bc1, bc2::DCBCT)        = true
fill_first(bc1::DCBCT, bc2::DCBCT) = true
fill_first(bc1::PBCT,  bc2)        = false
fill_first(bc1::MCBCT, bc2)        = false
fill_first(bc1::PBCT,  bc2::MCBCT) = true
fill_first(bc1::MCBCT, bc2::PBCT)  = false
fill_first(bc1, bc2::PBCT)         = true
fill_first(bc1, bc2::MCBCT)        = true
fill_first(bc1::PBCT,  bc2::PBCT)  = true
fill_first(bc1::MCBCT, bc2::MCBCT) = true
fill_first(bc1, bc2)               = true
