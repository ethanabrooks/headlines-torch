--
-- Created by IntelliJ IDEA.
-- User: Ethan
-- Date: 7/12/16
-- Time: 8:26 PM
-- To change this template use File | Settings | File Templates.
--
require 'nn'
require 'rnn'
require 'nngraph'


local batchSize = 2
local inSeqLen = 3 -- sequence length
local outSeqLen = 5 -- sequence length
local hiddenSize = 4
local s = torch.ones(batchSize, hiddenSize) + 1
local nIndex = 100

local input = nn.Identity()()
local sigmoid = nn.gModule({input}, {nn.Sigmoid()(input)})

local r = nn.Recurrent(
    nn.Identity(),
    nn.LookupTable(nIndex, hiddenSize),
    nn.Linear(hiddenSize, hiddenSize),
    sigmoid,
    inSeqLen
)


local h = torch.range(1, inSeqLen * batchSize * hiddenSize)
:resize(inSeqLen, batchSize, hiddenSize)
local s = torch.range(1, batchSize * hiddenSize)
:resize(batchSize, hiddenSize) + 1
local y = torch.range(1, batchSize * hiddenSize)
:resize(batchSize, hiddenSize) + 1
print(r.transferModule:sharedClone())
r:forward(y)
