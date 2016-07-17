--
-- Created by IntelliJ IDEA.
-- User: Ethan
-- Date: 7/16/16
-- Time: 9:22 AM
-- To change this template use File | Settings | File Templates.
--
require 'nn'


-- hyper-parameters
local batchSize = 3
local inSeqLen = 5
local outSeqLen = 4
local hiddenSize = 2
local nIndex = 100

local zipRepeating = nn.ConcatTable()
for i = 1, outSeqLen do
    local parallel = nn.ParallelTable()
    parallel:add(nn.Identity()) -- repeating
    parallel:add(nn.SelectTable(i)) -- nonrepeating
    zipRepeating:add(parallel)
end

local h = torch.range(1, inSeqLen * batchSize * hiddenSize)
:resize(inSeqLen, batchSize, hiddenSize)
local y = torch.range(1, outSeqLen * batchSize * hiddenSize)
:resize(outSeqLen, batchSize, hiddenSize) + 3
local s = torch.range(1, batchSize * hiddenSize)
:resize(batchSize, hiddenSize) + 1
y = nn.SplitTable(1):forward(y)
out = zipRepeating:forward{h, y}
zipRepeating:backward({h, y}, out)
