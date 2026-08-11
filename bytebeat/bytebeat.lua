-- bytebeat.lua — one-file bytebeat renderer, pure Lua 5.3+, zero dependencies.
--
-- A bytebeat is a song defined by a single integer formula over a sample
-- counter t; the low 8 bits of f(t) become the waveform. This script
-- evaluates such a formula per sample and writes an 8-bit unsigned mono WAV.
--
-- Usage:
--   lua bytebeat.lua "t*(t>>10&42)" out.wav --seconds 30 --rate 8000
--   lua bytebeat.lua --preset 1 out.wav [--seconds 10] [--rate 8000]
--
-- Formulas use C-style operators. Lua 5.3's native bitwise ops (& | ~ << >>)
-- share C's precedence, so classic formulas paste in unchanged, with two
-- adaptations handled automatically:
--   ^  (C xor)      -> ~  (Lua xor)
--   /  (C int div)  -> // (Lua floor div)
-- Formulas run in an empty sandbox — only t and operators are available.

local PRESETS = {
  "t*(42&t>>10)",                      -- 1: the "42 melody"
  "(t>>6|t<<4)+(t>>6|t>>8)",           -- 2: stepped arpeggio drone
  "t*((t>>12|t>>8)&63&t>>4)",          -- 3: classic viznut jam
  "t*(t>>11&t>>8&123&t>>3)",           -- 4: dark marching rhythm
  "t*((t>>9|t>>13)&15)",               -- 5: bright cyclic bleeps
}

local USAGE = [[
usage:
  lua bytebeat.lua "formula" out.wav [--seconds N] [--rate N]
  lua bytebeat.lua --preset N out.wav [--seconds N] [--rate N]

options:
  --preset N   use built-in formula 1..%d (see below)
  --seconds N  length in seconds (default 10)
  --rate N     sample rate in Hz (default 8000)

presets:
%s
]]

-- Translate a C-style bytebeat formula into a Lua function of t.
local function compile(expr)
  local src = expr:gsub("%^", "~"):gsub("/", "//")
  local chunk, err = load("return function(t) return " .. src .. " end",
                          "bytebeat", "t", {})
  if not chunk then
    error(("invalid formula %q: %s"):format(expr, err), 0)
  end
  return chunk()
end

-- f(t) -> 8-bit unsigned sample. Bitwise ops need integers, so float
-- results (e.g. from a formula someone wrote with math outside the
-- sandbox) are floored first.
local function make_eval(fn)
  return function(t)
    local v = fn(t)
    if math.type(v) == "float" then
      v = math.floor(v)
    end
    return v & 0xFF
  end
end

-- 44-byte RIFF/WAVE header for 8-bit unsigned mono PCM.
local function wav_header(n_samples, rate)
  return table.concat({
    "RIFF", string.pack("<I4", 36 + n_samples), "WAVE",
    "fmt ", string.pack("<I4", 16),
    string.pack("<I2I2I4I4I2I2", 1, 1, rate, rate, 1, 8),
    "data", string.pack("<I4", n_samples),
  })
end

local function render(formula, path, seconds, rate)
  local eval = make_eval(compile(formula))
  local n = math.floor(seconds * rate + 0.5)
  if n <= 0 then error("nothing to render (seconds * rate == 0)", 0) end

  local f = assert(io.open(path, "wb"))
  f:write(wav_header(n, rate))
  local buf = {}
  for t = 0, n - 1 do
    buf[#buf + 1] = string.char(eval(t))
    if #buf == 8192 then
      f:write(table.concat(buf))
      buf = {}
    end
  end
  if #buf > 0 then f:write(table.concat(buf)) end
  f:close()
  return n
end

local function main(argv)
  local formula, out = nil, nil
  local seconds, rate = 10, 8000
  local i = 1
  while i <= #argv do
    local a = argv[i]
    if a == "--seconds" or a == "--rate" or a == "--preset" then
      local v = tonumber(argv[i + 1] or "")
      if not v then error(a .. " needs a numeric argument", 0) end
      if a == "--seconds" then seconds = v
      elseif a == "--rate" then rate = v
      else
        formula = PRESETS[v]
        if not formula then
          error(("no preset %d (have 1..%d)"):format(v, #PRESETS), 0)
        end
      end
      i = i + 2
    elseif a:sub(1, 2) == "--" then
      error("unknown option: " .. a, 0)
    elseif formula == nil and out == nil then
      formula = a
      i = i + 1
    elseif out == nil then
      out = a
      i = i + 1
    else
      error("unexpected argument: " .. a, 0)
    end
  end

  if not formula or not out then
    local list = {}
    for k, p in ipairs(PRESETS) do
      list[k] = ("  %d  %s"):format(k, p)
    end
    io.stderr:write(USAGE:format(#PRESETS, table.concat(list, "\n")))
    os.exit(1)
  end

  local n = render(formula, out, seconds, rate)
  print(("wrote %s: %.1fs @ %d Hz (%d samples)"):format(
        out, n / rate, rate, n))
end

local ok, err = pcall(main, arg)
if not ok then
  io.stderr:write("error: " .. tostring(err) .. "\n")
  os.exit(1)
end
