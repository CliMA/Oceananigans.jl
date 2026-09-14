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
Base.@constprop :aggressive function permute_boundary_conditions(bcs)

    split_x_halo_filling = split_halo_filling(bcs.west, bcs.east)
    split_y_halo_filling = split_halo_filling(bcs.south, bcs.north)

    # A single assignment keeps `sides` unboxed in the closures below
    sides, bcs_tuple = if split_x_halo_filling && split_y_halo_filling
        (West(), East(), South(), North(), BottomAndTop()), (bcs.west, bcs.east, bcs.south, bcs.north, bcs.bottom)
    elseif split_x_halo_filling
        (West(), East(), SouthAndNorth(), BottomAndTop()), (bcs.west, bcs.east, bcs.south, bcs.bottom)
    elseif split_y_halo_filling
        (WestAndEast(), South(), North(), BottomAndTop()), (bcs.west, bcs.south, bcs.north, bcs.bottom)
    else
        (WestAndEast(), SouthAndNorth(), BottomAndTop()), (bcs.west, bcs.south, bcs.bottom)
    end

    perm = filling_order(map(fill_priority, bcs_tuple))

    ordered_sides = ntuple(Val(length(sides))) do n
        Base.@_inline_meta
        @inbounds sides[perm[n]]
    end

    boundary_conditions = map(side -> extract_bc(bcs, side), ordered_sides)

    return ordered_sides, boundary_conditions
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
const OBCTC = Union{NFBC, Tuple{Vararg{NFBC}}}

# Distributed halos have to be filled last to allow the
# possibility of asynchronous communication:
# If other halos are filled after we initiate the distributed communication,
# (but before communication is completed) the halos will be overwritten.
# For this reason we always want to perform local halo filling first and then
# initiate communication

# Periodic is handled after Flux, Value, Gradient because
# Periodic fills also corners while Flux, Value, Gradient do not
# TODO: remove this ordering requirement (see issue https://github.com/CliMA/Oceananigans.jl/issues/3342)

# Order of halo filling (see `fill_priority`)
# 0) Nothing / no-op (Face on Bounded axis — no halo needed)
# 1) Flux, Value, Gradient (TODO: remove these BC and apply them as fluxes)
# 2) Periodic (PBCT)
# 3) Shared Communication (MCBCT)
# 4) Distributed Communication (DCBCT)

# Sides are filled by increasing priority; among equal priorities, the side listed last goes first.
# Everything here folds at compile time because the priorities depend only on the boundary condition types.
@inline fill_priority(::Nothing) = 0
@inline fill_priority(bc)        = 1
@inline fill_priority(::PBCT)    = 2
@inline fill_priority(::MCBCT)   = 3
@inline fill_priority(::DCBCT)   = 4

@inline fills_before((p₁, i₁), (p₂, i₂)) = p₁ < p₂ || (p₁ == p₂ && i₁ > i₂)

@inline insert_in_order(x, ::Tuple{}) = (x,)
Base.@constprop :aggressive @inline insert_in_order(x, sorted::Tuple) =
    fills_before(x, first(sorted)) ? (x, sorted...) : (first(sorted), insert_in_order(x, Base.tail(sorted))...)

@inline sort_in_order(::Tuple{}) = ()
Base.@constprop :aggressive @inline sort_in_order(t::Tuple) = insert_in_order(first(t), sort_in_order(Base.tail(t)))

Base.@constprop :aggressive @inline function filling_order(priorities::NTuple{N, Int}) where N
    keyed = ntuple(i -> (priorities[i], i), Val(N))
    return map(last, sort_in_order(keyed))
end
