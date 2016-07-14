--
-- Created by IntelliJ IDEA.
-- User: Ethan
-- Date: 7/10/16
-- Time: 9:59 AM
-- To change this template use File | Settings | File Templates.
--
require 'nn'

-- hyper-parameters
local batchSize = 1
local seqLen = 3 -- sequence length
local hiddenSize = 2
local nIndex = 100
local lr = 0.1
local x = torch.DoubleTensor(seqLen, batchSize, hiddenSize):fill(1)
print(x)

local feedback = nn.Sequential()
local parallel = nn.ConcatTable()
local embedMax = nn.Sequential()
feedback:add(parallel)
parallel:add(nn.Identity)
--parallel:add(embedMax)
parallel:add(nn.Identity)
print(parallel)
embedMax:add(nn.Max(3))
embedMax:add(nn.LookupTable(nIndex, hiddenSize))
feedback:add(nn.CAddTable())

print(feedback:forward(x))
