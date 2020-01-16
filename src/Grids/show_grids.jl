short_show(grid::RegularCartesianGrid{T}) where T =
    "RegularCartesianGrid{$T}(Nx=$(grid.Nx), Ny=$(grid.Ny), Nz=$(grid.Nz))"

show_domain(grid) = string("x ∈ [", grid.xF[1], ", ", grid.xF[end], "], ",
                           "y ∈ [", grid.yF[1], ", ", grid.yF[end], "], ",
                           "z ∈ [", grid.zF[1], ", ", grid.zF[end], "]")

show(io::IO, g::RegularCartesianGrid{T}) where T =
    print(io, "RegularCartesianGrid{$T}\n",
              "domain: x ∈ [$(g.xF[1]), $(g.xF[end])], y ∈ [$(g.yF[1]), $(g.yF[end])], z ∈ [$(g.zF[1]), $(g.zF[end])]", '\n',
              "  resolution (Nx, Ny, Nz) = ", (g.Nx, g.Ny, g.Nz), '\n',
              "   halo size (Hx, Hy, Hz) = ", (g.Hx, g.Hy, g.Hz), '\n',
              "grid spacing (Δx, Δy, Δz) = ", (g.Δx, g.Δy, g.Δz))

short_show(grid::VerticallyStretchedCartesianGrid{T}) where T =
    "VerticallyStretchedCartesianGrid{$T}(Nx=$(grid.Nx), Ny=$(grid.Ny), Nz=$(grid.Nz))"

show(io::IO, g::VerticallyStretchedCartesianGrid{T}) where T =
    print(io, "RegularCartesianGrid{$T}\n",
              "domain: x ∈ [$(g.xF[1]), $(g.xF[end])], y ∈ [$(g.yF[1]), $(g.yF[end])], z ∈ [$(g.zF[1]), $(g.zF[end])]", '\n',
              "  resolution (Nx, Ny, Nz) = ", (g.Nx, g.Ny, g.Nz), '\n',
              "   halo size (Hx, Hy, Hz) = ", (g.Hx, g.Hy, g.Hz), '\n')
