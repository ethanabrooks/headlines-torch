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
m = nn.GRU(hiddenSize, hiddenSize)
