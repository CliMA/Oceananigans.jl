using Oceananigans.Grids: AbstractGrid
using Oceananigans.Architectures: AbstractArchitecture

using KernelAbstractions: NoneEvent

import Oceananigans.BoundaryConditions:
    apply_x_bcs!,
    apply_y_bcs!,
    apply_z_bcs!

apply_x_bcs!(Gc, ::AbstractGrid, dep, c, ::HaloCommunicationBC, ::HaloCommunicationBC, args...) = NoneEvent()
apply_y_bcs!(Gc, ::AbstractGrid, dep, c, ::HaloCommunicationBC, ::HaloCommunicationBC, args...) = NoneEvent()
apply_z_bcs!(Gc, ::AbstractGrid, dep, c, ::HaloCommunicationBC, ::HaloCommunicationBC, args...) = NoneEvent()

