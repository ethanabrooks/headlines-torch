require 'rnn'
require 'nngraph'

-- hyper-parameters
local batchSize = 3
local seqLen = 4 -- sequence length
local hiddenSize = 2
local nIndex = 100
local lr = 0.1
local test = true

local x = torch.DoubleTensor(seqLen, batchSize, hiddenSize):fill(1)
test_input = torch.ones(batchSize, hiddenSize)

local input = nn.Identity()()
local output = nn.ArgMax(2, 2)(input)

local feedback = nn.Sequential()
local branch = nn.ConcatTable()
local embedMax = nn.Sequential()

feedback:add(branch)
branch:add(nn.Copy(nil, nil, true))
--branch:add(nn.Copy(nil, nil, true))
branch:add(embedMax)
--embedMax:add(nn.ArgMax(2, 2))
embedMax:add(nn.gModule({input}, {output}))
embedMax:add(nn.LookupTable(nIndex, hiddenSize))
feedback:add(nn.CAddTable())

-- build simple recurrent neural network
local r_train = nn.Recurrent(
    nn.Identity(),    -- start
    nn.Identity(),    -- input
    nn.Identity(),    -- feedback
    nn.GRU(hiddenSize, hiddenSize),         -- transfer
    seqLen,           -- rho
    nn.CAddTable() -- merge
)

local r_test = r_train:clone()
r_test.feedbackModule = feedback
r_test.mergeModule = nn.SelectTable(2)
r_test:buildRecurrentModule()

local rnn_test = nn.Sequencer(r_test, seqLen)
local rnn_train = nn.Sequencer(r_train, seqLen)

local rnn_forward = rnn_test:forward(x)
local rnn_forward = rnn_train:forward(x)
rnn_train:backward(x, x)
