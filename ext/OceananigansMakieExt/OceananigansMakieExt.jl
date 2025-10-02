module OceananigansMakieExt

include("DimensionalityUtils.jl")
include("MakieConversions.jl")
include("PlotExtensions.jl")
include("Imaginocean.jl")
include("CubedSphereVisualizations.jl")

using .DimensionalityUtils
using .MakieConversions
using .PlotExtensions
using .Imaginocean
using .CubedSphereVisualizations

export heatsphere!, heatlatlon!, specify_cubed_sphere_field_colorrange,
    specify_cubed_sphere_field_time_series_colorrange, panelwise_visualization, geo_heatmap_visualization,
    panelwise_visualization_animation, geo_heatmap_visualization_animation

end # module OceananigansMakieExt
