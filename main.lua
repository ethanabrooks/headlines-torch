--
-- Created by IntelliJ IDEA.
-- User: Ethan
-- Date: 7/9/16
-- Time: 11:41 AM
-- To change this template use File | Settings | File Templates.
--
require 'torch'
require 'rnn'
local decoderSelector = require 'decoder'

-- hyper-parameters
local batchSize = 3
local outSeqLen = 5 -- sequence length
local inSeqLen = 4
local hiddenSize = 2
local nIndex = 1000
local nClasses = 6
local depth = 3

local trainModel = nn.Sequential()
local deepGRU = nn.Sequential()
local encoder = nn.ParallelTable()
encoder:add(deepGRU)
encoder:add(nn.Identity())

deepGRU:add(nn.LookupTable(nIndex, hiddenSize))
for _ = 1, depth do
    deepGRU:add(nn.SeqGRU(hiddenSize, hiddenSize))
end

local zipRepeating = nn.ConcatTable()
for i = 1, outSeqLen do
    local parallel = nn.ParallelTable()
    parallel:add(nn.Identity()) -- repeating
    parallel:add(nn.Select(1, i)) -- nonrepeating
    zipRepeating:add(parallel)
end

local decoder = decoderSelector(true, inSeqLen, outSeqLen)
trainModel:add(encoder)
trainModel:add(zipRepeating)
trainModel:add(decoder)

local testModel = trainModel:sharedClone()
testModel.modules[3] = decoderSelector(false, inSeqLen, outSeqLen)

local x = torch.range(1, inSeqLen * batchSize)
:resize(inSeqLen, batchSize)
local s = torch.range(1, batchSize * hiddenSize)
:resize(batchSize, hiddenSize) + 1
local y = torch.range(1, outSeqLen * batchSize)
:resize(outSeqLen, batchSize) + 3

local out = trainModel:forward{ x, y }
trainModel:backward({ x, y }, out)
local out = testModel:forward{ x, y }
testModel:backward({ x, y }, out)

