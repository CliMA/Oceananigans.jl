using Adapt

using Oceananigans.Architectures: AbstractArchitecture
using Oceananigans.Grids: AbstractGrid
using Oceananigans.BoundaryConditions
using Oceananigans.BoundaryConditions: AbstractBoundaryConditionClassification

using KernelAbstractions: NoneEvent

import Base: show

import Oceananigans.BoundaryConditions: bc_str

import Oceananigans.BoundaryConditions:
    apply_x_bcs!,
    apply_y_bcs!,
    apply_x_east_bc!,
    apply_x_west_bc!,
    apply_y_south_bc!,
    apply_y_north_bc!

struct CubedSphereExchange <: AbstractBoundaryConditionClassification end

const CubedSphereExchangeBC = BoundaryCondition{<:CubedSphereExchange}

bc_str(::CubedSphereExchangeBC) ="CubedSphereExchange"

CubedSphereExchangeBoundaryCondition(val; kwargs...) = BoundaryCondition(CubedSphereExchange, val; kwargs...)

struct CubedSphereExchangeInformation{F, S}
    from_face :: F
      to_face :: F
    from_side :: S
      to_side :: S
end

CubedSphereExchangeInformation(; from_face, to_face, from_side, to_side) =
    CubedSphereExchangeInformation(from_face, to_face, from_side, to_side)

Base.show(io::IO, ex::CubedSphereExchangeInformation) =
    print(io, "CubedSphereExchangeInformation: (from: face $(ex.from_face) $(ex.from_side) side, to: face $(ex.to_face) $(ex.to_side) side)")

Base.summary(info::CubedSphereExchangeInformation) =
    "(from: face $(info.from_face) $(info.from_side) side, to: face $(info.to_face) $(info.to_side) side)"

function inject_cubed_sphere_exchange_boundary_conditions(field_bcs, face_number, face_connectivity)

    west_exchange_info = CubedSphereExchangeInformation(
        from_face = face_number,
        from_side = :west,
          to_face = face_connectivity[face_number].west.face,
          to_side = face_connectivity[face_number].west.side
    )

    east_exchange_info = CubedSphereExchangeInformation(
        from_face = face_number,
        from_side = :east,
          to_face = face_connectivity[face_number].east.face,
          to_side = face_connectivity[face_number].east.side
    )

    south_exchange_info = CubedSphereExchangeInformation(
        from_face = face_number,
        from_side = :south,
          to_face = face_connectivity[face_number].south.face,
          to_side = face_connectivity[face_number].south.side
    )

    north_exchange_info = CubedSphereExchangeInformation(
        from_face = face_number,
        from_side = :north,
          to_face = face_connectivity[face_number].north.face,
          to_side = face_connectivity[face_number].north.side
    )

    west_exchange_bc = CubedSphereExchangeBoundaryCondition(west_exchange_info)
    east_exchange_bc = CubedSphereExchangeBoundaryCondition(east_exchange_info)
    south_exchange_bc = CubedSphereExchangeBoundaryCondition(south_exchange_info)
    north_exchange_bc = CubedSphereExchangeBoundaryCondition(north_exchange_info)

    return FieldBoundaryConditions(west_exchange_bc,
                                   east_exchange_bc,
                                   south_exchange_bc,
                                   north_exchange_bc,
                                   field_bcs.bottom,
                                   field_bcs.top,
                                   field_bcs.immersed)
end

Adapt.adapt_structure(to, ::CubedSphereExchangeInformation) = nothing
Adapt.adapt_structure(to, ::CubedSphereExchangeBC) = nothing

# Don't "apply fluxes" across CubedSphere boundaries
@inline apply_x_east_bc!(  Gc, loc, ::CubedSphereExchangeBC, args...) = nothing
@inline apply_x_west_bc!(  Gc, loc, ::CubedSphereExchangeBC, args...) = nothing
@inline apply_y_north_bc!( Gc, loc, ::CubedSphereExchangeBC, args...) = nothing
@inline apply_y_south_bc!( Gc, loc, ::CubedSphereExchangeBC, args...) = nothing

apply_x_bcs!(Gc, ::AbstractGrid, c, ::CubedSphereExchangeBC, ::CubedSphereExchangeBC, ::AbstractArchitecture, args...) = NoneEvent()
apply_y_bcs!(Gc, ::AbstractGrid, c, ::CubedSphereExchangeBC, ::CubedSphereExchangeBC, ::AbstractArchitecture, args...) = NoneEvent()
