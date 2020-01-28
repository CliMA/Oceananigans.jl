#####
##### Convenient definitions
#####

"""
    second

A `Float64` constant equal to 1.0. Useful for increasing the clarity of scripts, e.g. `Δt = 1second`.
"""
const second = 1.0

"""
    minute

A `Float64` constant equal to 60`second`. Useful for increasing the clarity of scripts, e.g. `Δt = 15minute`.
"""
const minute = 60second

"""
    hour

A `Float64` constant equal to 60`minute`. Useful for increasing the clarity of scripts, e.g. `Δt = 3hour`.
"""
const hour   = 60minute

"""
    day

A `Float64` constant equal to 24`hour`. Useful for increasing the clarity of scripts, e.g. `Δt = 0.5day`.
"""
const day    = 24hour

"""
    meter

A `Float64` constant equal to 1.0. Useful for increasing the clarity of scripts, e.g. `Lx = 100meter`.
"""
const meter = 1.0

"""
    kilometer

A `Float64` constant equal to 1000`meter`. Useful for increasing the clarity of scripts, e.g. `Lx = 250kilometer`.
"""
const kilometer = 1000meter

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
