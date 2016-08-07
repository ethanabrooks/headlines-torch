--
-- Created by IntelliJ IDEA.
-- User: Ethan
-- Date: 7/10/16
-- Time: 9:59 AM
-- To change this template use File | Settings | File Templates.
--
require 'nn'
require 'nngraph'
require 'rnn'
require 'dpnn'

-- hyper-parameters
local batchSize = 2
local outSeqLen = 3 -- sequence length
local inSeqLen = 4
local hiddenSize = 5
local vocSize = 1000
local nClasses = 6
local depth = 3

local x_ = torch.range(1, inSeqLen * batchSize)
:resize(batchSize, inSeqLen)
local h_ = torch.range(1, inSeqLen * batchSize * hiddenSize)
:resize(batchSize, inSeqLen, hiddenSize)
local y_ = torch.range(1, outSeqLen * batchSize * nClasses)
:resize(batchSize, outSeqLen, nClasses)

--local oneHot = nn.OneHot(nClasses):forward(y_)
--print(oneHot)
local split = nn.SplitTable(2):forward(y_)
--print(split[1])
local zipRepeat = nn.ZipTableOneToMany():forward{h_, split }
--print(zipRepeat[1][1])
--print(zipRepeat[1][2])


local y = nn.SplitTable(2)()
local h        = nn.Identity()()
local output   = nn.ZipTableOneToMany(){h, y}
local model = nn.gModule({h, y}, {output})
--print(model:forward(y_))

print(model:forward{y_, h_}[1][1])
print(model:forward{y_, h_}[1][2])
print(model:forward{y_, h_}[2][1])


