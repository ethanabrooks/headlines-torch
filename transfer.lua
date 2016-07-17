--
-- Created by IntelliJ IDEA.
-- User: Ethan
-- Date: 7/11/16
-- Time: 9:23 PM
-- To change this template use File | Settings | File Templates.
--
require 'rnn'
require 'nngraph'

-- hyper-parameters
batchSize = 3
inSeqLen =4 -- sequence length
hiddenSize = 2
depth = 1

local Transfer, Parent = torch.class('nn.Transfer', 'nn.Module')

function Transfer:__init()
    Parent.__init(self)
    local make2d     = nn.Reshape(batchSize * inSeqLen, hiddenSize, false)
    local h, s       = nn.Identity()(), nn.Replicate(inSeqLen, 1)() -- both inSeqLen, batchSize, hiddenSize
    local h2d, s2d   = make2d(h), make2d(s) -- both inSeqLen * batchSize, hiddenSize
    local attention  = nn.CosineDistance(){h2d, s2d} -- inSeqLen * batchSize
    local broadcast  = nn.Replicate(hiddenSize, 2)(attention) -- inSeqLen * batchSize, hiddenSize
    local hWeighted  = nn.CMulTable(){h2d, broadcast} -- inSeqLen * batchSize, hiddenSize
    local hOrigShape = nn.Reshape(inSeqLen, batchSize, hiddenSize, false)(hWeighted) -- inSeqLen, batchSize, hiddenSize
    local gruInput   = nn.Sum(1)(hOrigShape) -- batchSize, hiddenSize
    local deepGru    = nn.Sequential()
    for _ = 1, depth do
        deepGru:add(nn.GRU(hiddenSize, hiddenSize))
    end
    self.net = nn.gModule({h, s},
                          {deepGru(gruInput)}) -- batchSize, hiddenSize
end

function Transfer:updateOutput(input)
    assert(_G.memory, "memory is nil")
    self.output = self.net:updateOutput{_G.memory, input}
    return self.output
end

function Transfer:updateGradInput(input, gradOutput)
    assert(_G.memory, "memory is nil")
    self.gradInput = self.net:updateGradInput({_G.memory, input}, gradOutput)
    return self.gradInput
end

function Transfer:accGradParameters(input, gradOutput)
    assert(_G.memory, "memory is nil")
    self.net:accGradParameters({_G.memory, input}, gradOutput)
end

--local h = torch.range(1, inSeqLen * batchSize * hiddenSize)
--:resize(inSeqLen, batchSize, hiddenSize)
--local s = torch.range(1, batchSize * hiddenSize)
--:resize(batchSize, hiddenSize) + 1
--local net = nn.Transfer()
--local out = net:forward(s)
--print(out)
--print(net:backward(s, out)[2])
