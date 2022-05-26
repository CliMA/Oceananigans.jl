module Units

export second, minute, hour, day, year, meter, kilometer,
       seconds, minutes, hours, days, years, meters, kilometers,
       KiB, MiB, GiB, TiB

#####
##### Convenient definitions
#####

"""
    second

A `Float64` constant equal to 1.0. Useful for increasing the clarity of scripts, e.g. `Δt = 1second`.
"""
const second = 1.0

"""
    seconds

A `Float64` constant equal to 1.0. Useful for increasing the clarity of scripts, e.g. `Δt = 7seconds`.
"""
const seconds = second

"""
    minute

A `Float64` constant equal to 60`seconds`. Useful for increasing the clarity of scripts, e.g. `Δt = 1minute`.
"""
const minute = 60seconds

"""
    minutes

A `Float64` constant equal to 60`seconds`. Useful for increasing the clarity of scripts, e.g. `Δt = 15minutes`.
"""
const minutes = minute

"""
    hour

A `Float64` constant equal to 60`minutes`. Useful for increasing the clarity of scripts, e.g. `Δt = 1hour`.
"""
const hour = 60minutes

"""
    hours

A `Float64` constant equal to 60`minutes`. Useful for increasing the clarity of scripts, e.g. `Δt = 3hours`.
"""
const hours = hour

"""
    day

A `Float64` constant equal to 24`hours`. Useful for increasing the clarity of scripts, e.g. `stop_time = 1day`.
"""
const day = 24hours

"""
    days

A `Float64` constant equal to 24`hours`. Useful for increasing the clarity of scripts, e.g. `stop_time = 7days`.
"""
const days = day

"""
    year

A `Float64` constant equal to 365`days`. Useful for increasing the clarity of scripts, e.g. `stop_time = 1year`.
"""
const year = 365days

"""
    years

A `Float64` constant equal to 365`days`. Useful for increasing the clarity of scripts, e.g. `stop_time = 100years`.
"""
const years = year

"""
    meter

A `Float64` constant equal to 1.0. Useful for increasing the clarity of scripts, e.g. `Lx = 1meter`.
"""
const meter = 1.0

"""
    meters

A `Float64` constant equal to 1.0. Useful for increasing the clarity of scripts, e.g. `Lx = 50meters`.
"""
const meters = meter

"""
    kilometer

A `Float64` constant equal to 1000`meters`. Useful for increasing the clarity of scripts, e.g. `Lx = 1kilometer`.
"""
const kilometer = 1000meters

"""
    kilometers

A `Float64` constant equal to 1000`meters`. Useful for increasing the clarity of scripts, e.g. `Lx = 5000kilometers`.
"""
const kilometers = kilometer

"""
    KiB

A `Float64` constant equal to 1024.0. Useful for increasing the clarity of scripts, e.g. `max_filesize = 250KiB`.
"""
const KiB = 1024.0

"""
    MiB

A `Float64` constant equal to 1024`KiB`. Useful for increasing the clarity of scripts, e.g. `max_filesize = 100MiB`.
"""
const MiB = 1024KiB

"""
    GiB

A `Float64` constant equal to 1024`MiB`. Useful for increasing the clarity of scripts, e.g. `max_filesize = 50GiB`.
"""
const GiB = 1024MiB

"""
    TiB

A `Float64` constant equal to 1024`GiB`. Useful for increasing the clarity of scripts, e.g. `max_filesize = 2TiB`.
"""
const TiB = 1024GiB

end # module
