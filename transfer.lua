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
local batchSize = 4
local inSeqLen = 2 -- sequence length
local outSeqLen = 5 -- sequence length
local hiddenSize = 3

local h = torch.range(1, inSeqLen * batchSize * hiddenSize)
    :resize(inSeqLen, batchSize, hiddenSize)
local s = torch.range(1, batchSize * hiddenSize)
    :resize(batchSize, hiddenSize) + 1
--print(s)

local input_layers = { nn.Sequential(), nn.Sequential() }
input_layers[2]:add(nn.Replicate(inSeqLen, 1)) -- replicate s x inSeqLen for comparison with h
local input = {}
for i = 1, 2 do
     -- combine batch and seq into one dimension
    input_layers[i]:add(nn.Reshape(batchSize * inSeqLen, hiddenSize))
     -- convert Sequential modules into nngraph nodes
    input[i] = input_layers[i]() -- {batchSize * inSeqLen x hiddenSize,
                                 --  batchSize * inSeqLen x hiddenSize}
end
local output = { nn.CosineDistance()(input) }
--local output = { nn.Mean(2)(nn.CSubTable()(input)) }
local align = nn.gModule(input, output)
local attention = align:forward{ h, s }

local input_layers = { nn.Sequential(), nn.Sequential() }
input_layers[2]:add(nn.Replicate(inSeqLen, 1)) -- replicate s x inSeqLen for comparison with h
local input = {}
for i = 1, 2 do
    -- combine batch and seq into one dimension
    input_layers[i]:add(nn.Reshape(batchSize * inSeqLen, hiddenSize))
    -- convert Sequential modules into nngraph nodes
    input[i] = input_layers[i]() -- {batchSize * inSeqLen x hiddenSize,
    --  batchSize * inSeqLen x hiddenSize}
end
local output = { nn.CosineDistance()(input) }
--local output = { nn.Mean(2)(nn.CSubTable()(input)) }
local align = nn.gModule(input, output)
local attention = align:forward{ h, s }

input = {
    nn.Reshape(batchSize * inSeqLen, hiddenSize)(), -- memory
    nn.Replicate(hiddenSize, 2)() -- attention
}
local attention_on_h = nn.CMulTable()(input)
local reshape_to_original = nn.Reshape(inSeqLen, batchSize, hiddenSize)
local weighted_h = reshape_to_original(attention_on_h) -- inSeqLen x batchSize x hiddenSize
local dot_on_h = nn.Sum(1)(weighted_h)

local transfer = nn.gModule(input, {dot_on_h})
print(transfer:forward{h, attention})
--print(transfer:forward{h, attention}[2])

