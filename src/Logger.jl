module Logger

export OceananigansLogger

using Dates
using Logging
using Crayons

import Logging: shouldlog, min_enabled_level, catch_exceptions, handle_message

const RED    = Crayon(foreground=:red)
const YELLOW = Crayon(foreground=:light_yellow)
const CYAN   = Crayon(foreground=:cyan)
const BLUE   = Crayon(foreground=:blue)

const BOLD      = Crayon(bold=true)
const UNDERLINE = Crayon(underline=true)

struct OceananigansLogger <: Logging.AbstractLogger
              stream :: IO
           min_level :: Logging.LogLevel
      message_limits :: Dict{Any,Int}
    show_info_source :: Bool
end

"""
    OceananigansLogger(stream::IO=stdout, level=Logging.Info; show_info_source=false)

Based on Logging.SimpleLogger, it tries to log all messages in the following format:

    [yyyy/mm/dd HH:MM:SS.sss] log_level message [-@-> source_file:line_number]

where the source of the message between the square brackets is included only if
`show_info_source=true` or if the message is not an info level message.
"""
OceananigansLogger(stream::IO=stdout, level=Logging.Info; show_info_source=false) =
    OceananigansLogger(stream, level, Dict{Any,Int}(), show_info_source)

shouldlog(logger::OceananigansLogger, level, _module, group, id) =
    get(logger.message_limits, id, 1) > 0

min_enabled_level(logger::OceananigansLogger) = logger.min_level

catch_exceptions(logger::OceananigansLogger) = false

function level_to_string(level)
    level == Logging.Error && return "ERROR"
    level == Logging.Warn  && return "WARN "
    level == Logging.Info  && return "INFO "
    level == Logging.Debug && return "DEBUG"
    return string(level)
end

function level_to_crayon(level)
    level == Logging.Error && return RED
    level == Logging.Warn  && return YELLOW
    level == Logging.Info  && return CYAN
    level == Logging.Debug && return BLUE
    return identity
end

function handle_message(logger::OceananigansLogger, level, message, _module, group, id,
                                filepath, line; maxlog = nothing, kwargs...)

    if !isnothing(maxlog) && maxlog isa Int
        remaining = get!(logger.message_limits, id, maxlog)
        logger.message_limits[id] = remaining - 1
        remaining > 0 || return nothing
    end

    buf = IOBuffer()
    iob = IOContext(buf, logger.stream)

    level_name = level_to_string(level)
    crayon = level_to_crayon(level)

    module_name = something(_module, "nothing")
    file_name   = something(filepath, "nothing")
    line_number = something(line, "nothing")
    msg_timestamp = Dates.format(Dates.now(), "[yyyy/mm/dd HH:MM:SS.sss]")

    formatted_message = "$(crayon(msg_timestamp)) $(BOLD(crayon(level_name))) $message"
    if logger.show_info_source || level != Logging.Info
        formatted_message *= " $(BOLD(crayon("-@->"))) $(UNDERLINE("$file_name:$line_number"))"
    end

    println(iob, formatted_message)
    write(logger.stream, take!(buf))

    return nothing
end

end # module
