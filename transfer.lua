--
-- Created by IntelliJ IDEA.
-- User: Ethan
-- Date: 7/11/16
-- Time: 9:23 PM
-- To change this template use File | Settings | File Templates.
--
require 'rnn'
require 'nngraph'

-- hyper-parameters
local batchSize = 2
local inSeqLen = 3 -- sequence length
local outSeqLen = 5 -- sequence length
local hiddenSize = 4
local depth = 1
local nIndex = 100
local nClasses = 7

local make2d  = nn.Reshape(batchSize * inSeqLen, hiddenSize, false)
local combine = nn.Sequential()
combine:add(nn.CAddTable())
combine:add(nn.Replicate(inSeqLen, 1))
combine:add(make2d)

local y, s      = nn.LookupTable(nIndex, hiddenSize)(), nn.Identity()() -- both batchSize, hiddenSize
local h, s_     = make2d(), combine{ y, s } -- both inSeqLen * batchSize, hiddenSize
local attention = nn.Replicate(hiddenSize, 2)(nn.CosineDistance(){ h, s_ }) -- inSeqLen * batchSize, hiddenSize

local process = nn.Sequential()
-- dot product
process:add(nn.CMulTable())
process:add(nn.Reshape(inSeqLen, batchSize, hiddenSize, false))
process:add(nn.Sum(1))
--deep GRU
for _ = 1, depth do
    process:add(nn.GRU(hiddenSize, hiddenSize))
end

local gruOutput = process{ h, attention }
local y_pred = nn.Linear(hiddenSize, nClasses)(gruOutput)
local transfer = nn.gModule({ h, y, s },
    { gruOutput, y_pred }) -- batchSize, hiddenSize

local h = torch.range(1, inSeqLen * batchSize * hiddenSize)
:resize(inSeqLen, batchSize, hiddenSize)
local y = torch.range(1, batchSize)
local s = torch.range(1, batchSize * hiddenSize)
:resize(batchSize, hiddenSize) + 1
local out = transfer:forward{h, y, s }
--out:backward(s, out)
