extract_bc(bcs, ::West)   = tuple(bcs.west)
extract_bc(bcs, ::East)   = tuple(bcs.east)
extract_bc(bcs, ::South)  = tuple(bcs.south)
extract_bc(bcs, ::North)  = tuple(bcs.north)
extract_bc(bcs, ::Bottom) = tuple(bcs.bottom)
extract_bc(bcs, ::Top)    = tuple(bcs.top)

extract_bc(bcs, ::BottomAndTop)  = (bcs.bottom, bcs.top)
extract_bc(bcs, ::WestAndEast)   = (bcs.west, bcs.east)
extract_bc(bcs, ::SouthAndNorth) = (bcs.south, bcs.north)

# In case of a DistributedCommunication paired with a
# Flux, Value or Gradient boundary condition, we split the direction in two single-sided
# fill_halo! events (see issue #3342)
# `permute_boundary_conditions` returns a 2-tuple containing the ordered operations to execute in
# position [1] and the associated boundary conditions in position [2]
function permute_boundary_conditions(bcs)

    split_x_halo_filling = split_halo_filling(bcs.west, bcs.east)
    split_y_halo_filling = split_halo_filling(bcs.south, bcs.north)

    if split_x_halo_filling
        if split_y_halo_filling
            sides      = [West(), East(), South(), North(), BottomAndTop()]
            bcs_array  = [bcs.west, bcs.east, bcs.south, bcs.north, bcs.bottom]
        else
            sides     = [West(), East(), SouthAndNorth(), BottomAndTop()]
            bcs_array = [bcs.west, bcs.east, bcs.south, bcs.bottom]
        end
    else
        if split_y_halo_filling
            sides     = [WestAndEast(), South(), North(), BottomAndTop()]
            bcs_array = [bcs.west, bcs.south, bcs.north, bcs.bottom]
        else
            sides     = [WestAndEast(), SouthAndNorth(), BottomAndTop()]
            bcs_array = [bcs.west, bcs.south, bcs.bottom]
        end
    end

    perm  = sortperm(bcs_array, lt=fill_first)
    sides = tuple(sides[perm]...)

    boundary_conditions = Tuple(extract_bc(bcs, side) for side in sides)

    return sides, boundary_conditions
end

side_name(::West) = :west
side_name(::East) = :east
side_name(::South) = :south
side_name(::North) = :north
side_name(::Bottom) = :bottom
side_name(::Top) = :top
side_name(::WestAndEast) = :west_and_east
side_name(::SouthAndNorth) = :south_and_north
side_name(::BottomAndTop) = :bottom_and_top

# Split direction in two distinct fill_halo! events in case of a communication boundary condition
# (distributed DCBC), paired with a Flux, Value or Gradient boundary condition
split_halo_filling(bcs1, bcs2)     = false
split_halo_filling(::DCBC, ::DCBC) = false
split_halo_filling(bcs1, ::DCBC)   = true
split_halo_filling(::DCBC, bcs2)   = true

# Same thing for MultiRegion boundary conditions
split_halo_filling(::MCBC, ::MCBC) = false
split_halo_filling(bcs1, ::MCBC)   = true
split_halo_filling(::MCBC, bcs2)   = true

# heterogenous distribute-shared communication is not supported
# TODO: support heterogeneous distributed-shared communication
split_halo_filling(::MCBC, ::DCBC) = throw("Cannot split MultiRegion and Distributed boundary conditions.")
split_halo_filling(::DCBC, ::MCBC) = throw("Cannot split MultiRegion and Distributed boundary conditions.")

#####
##### Halo filling order
#####

const PBCT  = Union{PBC,  Tuple{Vararg{PBC}}}
const MCBCT = Union{MCBC, Tuple{Vararg{MCBC}}}
const DCBCT = Union{DCBC, Tuple{Vararg{DCBC}}}
const OBCTC = Union{OBC,  Tuple{Vararg{OBC}}}

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
