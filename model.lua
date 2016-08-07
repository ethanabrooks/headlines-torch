--
-- Created by IntelliJ IDEA.
-- User: Ethan
-- Date: 7/9/16
-- Time: 11:41 AM
-- To change this template use File | Settings | File Templates.
--

require 'rnn'
local buildDecoder = require 'decoder'


local Seq2Seq, Parent = torch.class('nn.Seq2Seq', 'nn.Module')

function Seq2Seq:__init(cuda, train, hiddenSize, vocSize, depth, nClasses)

    function self.maybeCuda(module)
        if cuda then
            return module:cuda()
        else
            return module
        end
    end

    self.train = train
    self.hiddenSize = hiddenSize
    local model  = nn.Sequential()
    local decGRU = nn.Sequential()

    --- build encoder
    local encGRU  = nn.Sequential()
    :add(nn.Transpose{1, 2}) -- [hiddenSize, batchSize]
    :add(nn.LookupTable(vocSize, hiddenSize)) -- embed x
    for i = 1, depth do
        encGRU:add(nn.SeqGRU(hiddenSize, hiddenSize))
        local inputSize = i == 1 and hiddenSize * 2 + nClasses or hiddenSize
        decGRU:add(nn.GRU(inputSize, hiddenSize))
    end
    encGRU:add(nn.Transpose{1, 2})
    local encoder = nn.ParallelTable()
    :add(encGRU)
    :add(nn.OneHot(nClasses)) -- pass along y

    --- build weighted layers of decoder
    self.decGRU   = self.maybeCuda(decGRU)                              -- populated in next paragraph
    self.outLayer = self.maybeCuda(nn.Linear(hiddenSize, nClasses))     -- map hidden state to class preds
    self.weightedModules = {nn.LookupTable(vocSize, hiddenSize), self.decGRU, self.outLayer}

    model:add(encoder)
    model:add(buildDecoder(train, hiddenSize, 1, 1, self.weightedModules))
    self.model = self.maybeCuda(model)
    Parent.__init(self)
end

function Seq2Seq:updateOutput(input)
    local x, y = unpack(input)
    local inSeqLen, outSeqLen = x:size(2), y:size(2)
    if self.inSeqLen ~= inSeqLen or self.outSeqLen ~= outSeqLen then
        self.inSeqLen, self.outSeqLen = inSeqLen, outSeqLen
        print('new decoder')

        -- clone weighted modules
        local weightedModules = {}
        for i = 1, #self.weightedModules do
            weightedModules[i] = self.weightedModules[i]:sharedClone()
        end

        -- build decoder
        self.model.modules[2] = self.maybeCuda(
            -- function creates a new decoder layer
            buildDecoder(self.train, self.hiddenSize,
                         inSeqLen, outSeqLen, weightedModules)
        )
    end
    self.output = self.model:forward(input)
    return self.output
end

function Seq2Seq:training()
    self.inSeqLen, self.outSeqLen = nil, nil
    Parent.training(self)
end

function Seq2Seq:evaluate()
    self.inSeqLen, self.outSeqLen = nil, nil
    Parent.evaluate(self)
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

function Seq2Seq:__tostring__()
    return self.model:__tostring__()
end

function test()
    -- hyper-parameters
    local batchSize = 2
    local inSeqLen = 3
    local outSeqLen = 4 -- sequence length
    local hiddenSize = 5
    local vocSize = 1000
    local nClasses = outSeqLen * batchSize * 2
    local depth = 3

    local x = torch.range(1, inSeqLen * batchSize)
    :resize(batchSize, inSeqLen)
    local y1 = torch.range(1, outSeqLen * batchSize)
    :resize(batchSize, outSeqLen) + 3
    local y2 = torch.range(1, outSeqLen * batchSize)
    :resize(batchSize, outSeqLen + 3) + 3

    local trainModel = nn.Seq2Seq(false, true, hiddenSize, vocSize, depth, nClasses)
--    local testModel = trainModel:sharedClone()
--    testModel.train = false
    local out = trainModel:forward{ x, y1 }
    local criterion = nn.SequencerCriterion(nn.CrossEntropyCriterion())
    local tgts = torch.ones(batchSize, outSeqLen):split(1, 2)
    print(criterion:forward(out, tgts))
--    local out = trainModel:forward{ x, y2 }
--    local gradOutput = trainModel:backward({ x, y1 }, out)
--    trainModel.model:updateParameters(1)
--    local out = trainModel:forward{ x, y1 }
--    local out = testModel:forward{ x, y2 }
--    testModel:backward({ x, y2 }, out)
end
test()