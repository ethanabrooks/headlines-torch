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

function f(fname)
    fname:find(5)
end

local string = "abc.csv"
for f in paths.files('train', "5") do
    print(f)
    end
