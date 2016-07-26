--
-- Created by IntelliJ IDEA.
-- User: Ethan
-- Date: 7/9/16
-- Time: 11:41 AM
-- To change this template use File | Settings | File Templates.
--
require 'torch'
require 'rnn'

-- hyper-parameters
local batchSize = 10
local outSeqLen = 5 -- sequence length
local inSeqLen = 4
local hiddenSize = 2
local vocSize = 1000
local nClasses = 6
local depth = 3


local Seq2Seq, Parent = torch.class('nn.Seq2Seq', 'nn.Module')

function Seq2Seq:__init(cuda, train, batchSize, hiddenSize, vocSize, depth, nClasses)

    self.cuda = cuda
    self.train = train
    self.batchSize = batchSize
    self.hiddenSize = hiddenSize
    self.model = nn.Sequential()

    --- build weighted layers of decoder
    local decGRU = nn.Sequential()
    self.decWeightedModules = {
        nn.LookupTable(vocSize, hiddenSize), -- embed y
        decGRU,                              -- populated in next paragraph
        nn.Linear(hiddenSize, nClasses),     -- map hidden state to class preds
    }

    --- build encoder
    local encoder = nn.ParallelTable()
    local encGRU  = nn.Sequential()
    encGRU:add(nn.LookupTable(vocSize, hiddenSize)) -- embed x
    for _ = 1, depth do
        encGRU:add(nn.SeqGRU(hiddenSize, hiddenSize))
        decGRU:add(nn.GRU(hiddenSize, hiddenSize))
    end
    encoder:add(encGRU)
    encoder:add(nn.Identity()) -- pass along y

    self.model:add(encoder)
    self.model:add(nn.Identity()) -- decoder placeholder
    if cuda then self.model:cuda() end
    self.decoders = {} -- memoize decoders
    Parent.__init(self)
end

function Seq2Seq:updateOutput(input)
    local x, y = unpack(input)
    local inSeqLen, outSeqLen = x:size(1), y:size(1)
    local key = { inSeqLen, outSeqLen, self.train }
    if not self.decoders[key] then

        -- clone weighted modules
        local weightedModules = {}
        for i = 1, #self.decWeightedModules do
            weightedModules[i] = self.decWeightedModules[i]:sharedClone()
        end

        -- memoize new decoder
        self.decoders[key] = require('decoder')(
            self.train, self.batchSize, self.hiddenSize,
            inSeqLen, outSeqLen, weightedModules
        ) -- function creates a new decoder layer
        if self.cuda then
            self.decoders[key]:cuda()
        end
    end
    self.model.modules[2] = self.decoders[key]
    self.output = self.model:forward(input)
    return self.output
end

function Seq2Seq:updateGradInput(input, gradOutput)
    self.gradInput = self.model:updateGradInput(input, gradOutput)
    return self.gradInput
end

function Seq2Seq:accGradParameters(input, gradOutput)
    self.model:accGradParameters(input, gradOutput)
end

function Seq2Seq:updateParameters(lr)
    self.model:updateParameters(lr)
end

function test()
    local x = torch.range(1, inSeqLen * batchSize)
    :resize(inSeqLen, batchSize)
    local y1 = torch.range(1, outSeqLen * batchSize)
    :resize(outSeqLen, batchSize) + 3
    local y2 = torch.range(1, outSeqLen * batchSize)
    :resize(outSeqLen + 3, batchSize) + 3

    local trainModel = nn.Seq2Seq(false, true, batchSize, hiddenSize, vocSize, depth, nClasses)
    local testModel = trainModel:sharedClone()
    testModel.train = false
    local out = trainModel:forward{ x, y1 }
    local out = trainModel:forward{ x, y2 }
    local criterion = nn.SequencerCriterion(nn.CrossEntropyCriterion())
    print(criterion:forward(out[1], torch.ones(batchSize)))
    local gradOutput = trainModel:backward({ x, y2 }, out)
    print(out[1])
    trainModel.model:updateParameters(1)
    local out = trainModel:forward{ x, y1 }
    local out = testModel:forward{ x, y2 }
    print(out[1])
    testModel:backward({ x, y2 }, out)
end
--test()
