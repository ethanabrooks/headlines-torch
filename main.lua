--
-- User: Ethan
-- Date: 7/17/16
-- Time: 6:31 PM
-- To change this template use File | Settings | File Templates.
--
require 'nn'
require 'optim'
require 'model'

--[[ command line arguments ]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train Seq2Seq on headlines')
cmd:text('Options:')
-- training
--cmd:option('--startlr', 0.05, 'learning rate at t=0')
--cmd:option('--minlr', 0.00001, 'minimum learning rate')
--cmd:option('--saturate', 400, 'epoch at which linear decayed LR will reach minlr')
--cmd:option('--schedule', '', 'learning rate schedule. e.g. {[5] = 0.004, [6] = 0.001}')
--cmd:option('--momentum', 0.9, 'momentum')
--cmd:option('--maxnormout', -1, 'max l2-norm of each layer\'s output neuron weights')
--cmd:option('--cutoff', -1, 'max l2-norm of concatenation of all gradParam tensors')


cmd:option('--batchSize', 32, 'number of examples per batch')
cmd:option('--cuda', true, 'use CUDA')
cmd:option('--device', 1, 'sets the device (GPU) to use')
cmd:option('--maxepoch', 1000, 'maximum number of epochs to run')
cmd:option('--earlystop', 50, 'maximum number of epochs to wait to find a better local minima for early-stopping')
cmd:option('--progress', false, 'print progress bar')
cmd:option('--silent', false, 'don\'t print anything to stdout')
cmd:option('--uniform', 0.1, 'initialize parameters using uniform distribution between -uniform and uniform. -1 means default initialization')
-- rnn layer
cmd:option('--lstm', false, 'use Long Short Term Memory (nn.LSTM instead of nn.Recurrent)')
cmd:option('--bn', false, 'use batch normalization. Only supported with --lstm')
cmd:option('--gru', false, 'use Gated Recurrent Units (nn.GRU instead of nn.Recurrent)')
cmd:option('--seqlen', 5, 'sequence length : back-propagate through time (BPTT) for this many time-steps')
cmd:option('--depth', 1, 'depth of GRU')
cmd:option('--inputsize', -1, 'size of lookup table embeddings. -1 defaults to hiddensize[1]')
cmd:option('--hiddenSize', 200, 'number of hidden units used at output of each recurrent layer. When more than one is specified, RNN/LSTMs/GRUs are stacked')
cmd:option('--dropout', 0, 'apply dropout with this probability after each rnn layer. dropout <= 0 disables it.')
-- data
cmd:option('--batchSize', 32, 'number of examples per batch')
cmd:option('--trainsize', -1, 'number of train examples seen between each epoch')
cmd:option('--validsize', -1, 'number of valid examples used for early stopping and cross-validation')
cmd:option('--savepath', paths.concat('main', 'rnnlm'), 'path to directory where experiment log (includes model) will be saved')
cmd:option('--id', '', 'id string of this experiment (used to name output file) (defaults to a unique id)')

cmd:text()
local opt = cmd:parse(arg or {})

local vocSize = 20000 -- TODO

local logger = optim.Logger('loss_log.txt')
model = nn.Seq2Seq(false, true, opt.batchSize, opt.hiddenSize, vocSize, opt.depth, vocSize)
criterion = nn.SequencerCriterion(nn.CrossEntropyCriterion()) -- todo make this compatible with model
x, dl_dx = model:getParameters()

-- TODO: make these for adagrad
optim_params = {
    learningRate = 1e-3,
    learningRateDecay = 1e-4,
    weightDecay = 0,
    momentum = 0
}

local feval = function(x_new)
    if x ~= x_new then
        x:copy(x_new)
    end

    -- reset gradients
    -- (gradients are always accumulated, to accommodate batch methods)
    dl_dx:zero()

    -- evaluate the loss function and its derivative wrt x, for that sample
    local loss_x = criterion:forward(model:forward(inputs), target)
    model:backward(inputs, criterion:backward(output, target))

    -- return loss(x) and d/dx(loss)
    return loss_x, dl_dx
end

-- run
for _ = 1, opt.maxepoch do
    local current_loss = 0
    for _, set in pairs({'train', 'test'}) do
        for batchDir in paths.iterdirs(set) do
            local batchPath = paths.concat(set, batchDir)

            -- TODO: batching
            inputs = torch.load(paths.concat(batchPath, 'article.dat'))
            target = torch.load(paths.concat(batchPath, 'title.dat'))
            inputs = {inputs, target}

            -- TODO: print stuff
            -- TODO: log stuff
            -- TODO: only make updates ever n intervals
            if set == 'train' then
                local _, fs = optim.sgd(feval, x, optim_params)
                current_loss = current_loss + fs[1]
            else
                local pred = model:forward(inputs)
                -- TODO: evaluate
            end
        end
    end

    print('current loss = ' .. current_loss)

    logger:add{['training error'] = current_loss}
    logger:style{['training error'] = '-'}
--    logger:plot()
end
