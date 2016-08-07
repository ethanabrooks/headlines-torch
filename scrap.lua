require 'rnn'
require 'nngraph'

-- hyper-parameters
local batchSize = 2
local inSeqLen = 3
local outSeqLen = 4 -- sequence length
local hiddenSize = 5
local vocSize = 1000
local nClasses = 16
local depth = 3

--local embed, deepGRU, outLayer = unpack(weightedModules)

local weighted = {nn.LookupTable(vocSize, hiddenSize), nn.GRU(hiddenSize * 2 + nClasses, hiddenSize), nn.Linear(hiddenSize, nClasses)}
local embed, deepGRU, outLayer = unpack(weighted)

--- build zipRepeating: {h, y} -> {{h, y1}, {h, y2}, ...}
local y            = nn.SplitTable(2)()
local h            = nn.Identity()()
local output       = nn.ZipTableOneToMany(){h, y}
local zipRepeating = nn.gModule({h, y}, {output})

--- {{y, h}, s(t-1)} -> A(h) + y + s(t-1)
local input = nn.Identity()()          -- {y, h}
local h     = nn.SelectTable(1)(input) -- [batchSize, inSeqLen, hiddenSize]
local y     = nn.SelectTable(2)(input) -- [batchSize, nClasses]
local s     = nn.Identity()()          -- [batchSize, hiddenSize]

local broadcast = nn.Replicate(inSeqLen, 2)(s)                 -- [batchSize, inSeqLen, hiddenSize]
local h2d       = nn.Reshape(-1, hiddenSize, false)(h)         -- [batchSize * inSeqLen, hiddenSize]
local s2d       = nn.Reshape(-1, hiddenSize, false)(broadcast) -- [batchSize * inSeqLen, hiddenSize]

--- {s, h} -> attention over h
------------------------------
--- s:      [inSeqLen * batchSize, hiddenSize]
--- h:      [inSeqLen * batchSize, hiddenSize]
--- output: [inSeqLen * batchSize, hiddenSize]
local align = nn.Sequential()
:add(nn.CosineDistance())              -- compare                   [inSeqLen * batchSize]
:add(nn.Reshape(-1, inSeqLen, false))  -- expose inSeqLen dimension [batchSize, inSeqLen]
:add(nn.SoftMax())                     -- softmax over inSeqLen dim [inSeqLen, batchSize]
:add(nn.Reshape(-1, false))            -- ravel                     [batchSize * inSeqLen]
:add(nn.Replicate(hiddenSize, 2))      -- broadcast                 [batchSize * inSeqLen, hiddenSize]
local attention = align{s2d, h2d}      --                           [batchSize * inSeqLen, hiddenSize]

--- {attention, h} -> attention(h)
----------------------------------
--- attention: [inSeqLen * batchSize, hiddenSize]
--- h:         [batchSize, hiddenSize]
--- output:    [batchSize, hiddenSize]
local apply = nn.Sequential()
-- dot product
:add(nn.CMulTable())                              -- [batchSize * inSeqLen, hiddenSize]
:add(nn.Reshape(-1, inSeqLen, hiddenSize, false)) -- [batchSize, inSeqLen, hiddenSize]
:add(nn.Sum(2))                                   -- [batchSize, hiddenSize]
local weighted_h = apply{attention, h2d}          -- [batchSize, hiddenSize]

--- {weighted_h, y, s} -> concat{weighted_h, y, s}
--------------------------------------------------
--- weighted_h: [batchSize, hiddenSize]
--- y:          [batchSize, hiddenSize]
--- s:          [batchSize, hiddenSize]
local concat = nn.Concat(2)
:add(nn.SelectTable(1))
:add(nn.SelectTable(2))
:add(nn.SelectTable(3))
local gruInput = concat{weighted_h, y, s} -- [batchSize, 2 * hiddenSize + nClasses]
local output   = deepGRU(gruInput)        -- [batchSize, hiddenSize]
local rm = nn.gModule({ input, s }, { output })

local sequencer = nn.Sequencer(
    nn.Sequential()
    :add(nn.Recurrence(rm, hiddenSize, 2, 1))
    :add(outLayer)
)
local decoder = nn.Sequential()
:add(zipRepeating)
:add(sequencer)

local h = torch.range(1, inSeqLen * batchSize * hiddenSize)
:resize(batchSize, inSeqLen, hiddenSize)
local y = torch.range(1, batchSize * nClasses * outSeqLen)
:resize(batchSize, outSeqLen, nClasses)
local s = torch.range(1, batchSize * hiddenSize)
:resize(batchSize, hiddenSize)
--print(y:size())
--print(h:size())
--print('all')
--print(decoder:forward{h, y})
--print('first')
--print(rm:forward{{h, y}, s}[1])
--print('second')
--print(rm:forward{{h, y}, s}[2])
--print('third')
--print(rm:forward{{h, y}, s}[3])
--print('fourth')
--print(rm:forward{{h, y}, s}[4])
--

