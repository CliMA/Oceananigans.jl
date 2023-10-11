
#####
##### Input file setup: has to be done only once!
#####

# Save a file named `filename`, containing a time series of surface heat flux and surface stress
# at times `times` on a grid `grid` 
function generate_input_data!(grid, times, filename; ρ_ocean = 1000, cp_ocean = 4000)

    Qˢ_tmp = Field((Center, Center, Nothing), grid)   
    τˣ_tmp = Field((Face, Center, Nothing), grid)   
    
    # Fictitious data
    Qₜ(y, t) = 200 * cos(2π * (y - 15) / 60) * sin(π * (t - 365days) / 365days) / ρ_ocean / cp_ocean
    τₜ(y, t) = 0.1 * cos(4π * (y - 15) / 60) * cos(π * (t - 365days) / 365days) / ρ_ocean 

    mask(y)  = -4 * ((y - 15) / 60) * ((y - 15) / 60 - 1)
    τᵢ(y, t) = τₜ(y, t) * mask(y)
    
    Qˢ = FieldTimeSeries((Center, Center, Nothing), grid, times; 
                         backend = OnDisk(),
                         path = filename,
                         name = "Qˢ")

    τˣ = FieldTimeSeries((Face, Center, Nothing), grid, times; 
                         backend = OnDisk(),
                         path = filename,
                         name = "τˣ")

    # write down the data on file
    for t in eachindex(times)
        @info "writing down data for time $t"
        set!(Qˢ_tmp, (x, y) -> Qₜ(y, times[t]))
        set!(τˣ_tmp, (x, y) -> τᵢ(y, times[t]))

        set!(Qˢ, Qˢ_tmp, t)
        set!(τˣ, τˣ_tmp, t)
    end
end