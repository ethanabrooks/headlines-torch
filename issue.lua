--
-- Created by IntelliJ IDEA.
-- User: Ethan
-- Date: 7/10/16
-- Time: 10:34 PM
-- To change this template use File | Settings | File Templates.
--

require 'rnn'
require 'nn'

-- hyper-parameters
local batchSize = 3
local hiddenSize = 2
local x = torch.ones(batchSize, hiddenSize)

--local feedback = nn.Sequential()
local concat = nn.ConcatTable()
--feedback:add(concat)
concat:add(nn.Copy(nil, nil, true))
concat:add(nn.Copy(nil, nil, true))
--feedback:add(nn.CAddTable())

local r = nn.Recurrent(
    nn.Identity(), -- start
    nn.Identity(), -- input
    nn.CAddTable(),
    concat -- transfer
)
print(r)
r:updateOutput(x)
r:updateOutput(x)
print(r:updateOutput(x))

