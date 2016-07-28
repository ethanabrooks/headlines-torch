require 'nn'

-- training
--cmd:option('--startlr', 0.05, 'learning rate at t=0')
--cmd:option('--minlr', 0.00001, 'minimum learning rate')
--cmd:option('--saturate', 400, 'epoch at which linear decayed LR will reach minlr')
--cmd:option('--schedule', '', 'learning rate schedule. e.g. {[5] = 0.004, [6] = 0.001}')
--cmd:option('--momentum', 0.9, 'momentum')
--cmd:option('--maxnormout', -1, 'max l2-norm of each layer\'s output neuron weights')
--cmd:option('--cutoff', -1, 'max l2-norm of concatenation of all gradParam tensors')

require 'cutorch'
require 'cunn'
require 'optim'
require 'model'
local depth = 1
local vocSize = 10
local hiddenSize = 3
local nClasses = 3

--local Test, Parent = torch.class('nn.Test', 'nn.Module')
--
--function Test:__init()
----    self.model = nn.Linear(2, 3):cuda()
--    self.model     = nn.Sequential()
--    local decGRU   = nn.Sequential()
--    self.yLookup   = nn.LookupTable(vocSize, hiddenSize):cuda() -- embed y
--    self.decGRU    = decGRU:cuda()                              -- populated in next paragraph
--    self.out_layer = nn.Linear(hiddenSize, nClasses):cuda()     -- map hidden state to class preds
--
----    local decWeightedModules = {self.yLookup, self.decGRU, self.out_layer}
--
----    for i, module in ipairs(decWeightedModules) do
----        decWeightedModules[i] = module:cuda()
----    end
--
--    --- build encoder
--    local encoder = nn.ParallelTable()
--    local encGRU  = nn.Sequential()
--    encGRU:add(nn.Transpose{1, 2}) -- [hiddenSize, batchSize]
--    encGRU:add(nn.LookupTable(vocSize, hiddenSize)) -- embed x
--    for _ = 1, depth do
--        encGRU:add(nn.SeqGRU(hiddenSize, hiddenSize))
----        decGRU:add(nn.GRU(hiddenSize, hiddenSize))
--    end
--    encoder:add(encGRU)
--    encoder:add(nn.Transpose{1, 2}) -- pass along y, but fix axes
--
--    self.model:add(encoder)
--    self.model:add(nn.Identity()) -- decoder placeholder
--    self.model:cuda()
--end
----local model = nn.Seq2Seq(true, true, 4, 10, 1, 100)
local model = nn.Seq2Seq(true, true, 2, 2, 2, 2)
print(model:getParameters())


