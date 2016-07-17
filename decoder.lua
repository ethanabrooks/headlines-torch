require 'rnn'
require 'nngraph'
require 'ZipRepeating'

-- hyper-parameters
local batchSize = 3
local seqLen = 4
local hiddenSize = 2
local nIndex = 100

_G.memory = torch.range(1, seqLen * batchSize * hiddenSize)
    :resize(seqLen, batchSize, hiddenSize)

require 'transfer'

local feedback = nn.Sequential()
local concat = nn.ConcatTable()
local embedMax = nn.Sequential()

-- build feedback
feedback:add(concat)
concat:add(nn.Copy(nil, nil, true))
concat:add(embedMax)
embedMax:add(nn.ArgMax(2, 2))
embedMax:add(nn.LookupTable(nIndex, hiddenSize))
feedback:add(nn.CAddTable())

local transfer = nn.Transfer()

-- build simple recurrent neural network
local r_train = nn.Recurrent(
    nn.Identity(), -- start
    nn.Identity(), -- input
    nn.Identity(), -- feedback
    transfer,
    seqLen,        -- rho
    nn.CAddTable() -- merge TODO
)

local r_test = r_train:clone()
r_test.feedbackModule = feedback -- TODO
r_test.mergeModule = nn.SelectTable(2) -- TODO
r_test:buildRecurrentModule()


local decode_train = nn.Sequencer(r_train, seqLen)
local decode_test = nn.Sequencer(r_test, seqLen)

local h = torch.range(1, seqLen * batchSize * hiddenSize)
:resize(inSeqLen, batchSize, hiddenSize)
local s = torch.range(1, batchSize * hiddenSize)
:resize(batchSize, hiddenSize) + 1
local y = torch.DoubleTensor(seqLen, batchSize, hiddenSize):fill(1)
--test_input = torch.ones(batchSize, hiddenSize)
local ziprepeating = ZipRepeating(seqLen)
local input = ziprepeating:forward{h, y }

local rnn_forward = decode_test:forward(input)
--print(rnn_forward)
local rnn_forward = decode_train:forward(input)
--print(rnn_forward)
--decode_train:backward(y, rnn_forward)