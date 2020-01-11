using Printf

"""
    pretty_filesize(s, suffix="B")

Convert a floating point value `s` representing a file size to a more human-friendly
formatted string with one decimal places with a `suffix` defaulting to "B". Depending on
the value of `s` the string will be formatted to show `s` using an SI prefix from bytes,
kiB (1024 bytes), MiB (1024² bytes), and so on up to YiB (1024⁸ bytes).
"""
function pretty_filesize(s, suffix="B")
    # Modified from: https://stackoverflow.com/a/1094933
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]
        abs(s) < 1024 && return @sprintf("%3.1f %s%s", s, unit, suffix)
        s /= 1024
    end
    return @sprintf("%.1f %s%s", s, "Yi", suffix)
end
