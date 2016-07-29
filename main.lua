--
-- User: Ethan
-- Date: 7/17/16
-- Time: 6:31 PM
-- To change this template use File | Settings | File Templates.
--
require 'nn'
require 'optim'
require 'model'
require 'rnn'
require 'cutorch'
require 'cunn'
cutorch.setDevice(3)

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


cmd:option('--device', 1, 'sets the device (GPU) to use')
cmd:option('--maxepoch', 200, 'maximum number of epochs to run')
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
cmd:option('--batchSize', 1024, 'number of examples per batch')
cmd:option('--trainsize', -1, 'number of train examples seen between each epoch')
cmd:option('--validsize', -1, 'number of valid examples used for early stopping and cross-validation')
cmd:option('--savepath', paths.concat('main', 'rnnlm'), 'path to directory where experiment log (includes model) will be saved')
cmd:option('--id', '', 'id string of this experiment (used to name output file) (defaults to a unique id)')

cmd:text()
local opt = cmd:parse(arg or {})

torch.setheaptracking(true)
local vocSize = 11000 -- TODO

local logger = optim.Logger('loss_log.txt')
local model = nn.Seq2Seq(true, true, opt.hiddenSize, vocSize, opt.depth, vocSize)
local criterion = nn.SequencerCriterion(nn.CrossEntropyCriterion()):cuda() -- todo make this compatible with model
local params, gradParams = model:getParameters()
local inputs, target

-- TODO: make these for adagrad
local optim_params = {
    learningRate = 1e-3,
    learningRateDecay = 1e-4,
    weightDecay = 0,
    momentum = 0
}

feval = function(x_new)
    if params ~= x_new then
        params:copy(x_new)
    end

    -- reset gradients
    -- (gradients are always accumulated, to accommodate batch methods)
    gradParams:zero()

    -- evaluate the loss function and its derivative wrt x, for that sample
    local target_table = target:split(1, 2)
    local outputs = model:forward(inputs)
    local loss_x = criterion:forward(outputs, target_table)
    model:backward(inputs, criterion:backward(outputs, target_table))

    -- return loss(x) and d/dx(loss)
    return loss_x, gradParams
end

-- run
for epoch = 1, opt.maxepoch do
    print('epoch', epoch)
    local loss = 0
    local instances_processed = 0
    for _, set in pairs({'train', 'test'}) do
        for batchDir in paths.iterdirs(set) do
            print(batchDir)
            local batchPath = paths.concat(set, batchDir)

            -- + 1 b/c lua is 1-indexed
            local bucket_inputs = torch.load(paths.concat(batchPath, 'article.dat')) + 1
            local bucket_target = torch.load(paths.concat(batchPath, 'title.dat')) + 1

            assert(bucket_inputs:size(1) == bucket_target:size(1))
            local num_batches = math.ceil(bucket_inputs:size(1) / opt.batchSize)
            for i = 1, num_batches do

                -- split bucket into batches
                local batch_start = (i - 1) * opt.batchSize + 1
                local batch_end = math.min(i * opt.batchSize + 1, bucket_inputs:size(1))
                inputs = bucket_inputs[{{batch_start, batch_end}}]:cuda()
                target = bucket_target[{{batch_start, batch_end}}]:cuda()
                inputs = {inputs, target}

                -- TODO: print stuff
                -- TODO: log stuff
                -- TODO: only make updates ever n intervals
                if set == 'train' then
                    if instances_processed % 1 == 1 then
                        local _, fs = optim.sgd(feval, params, optim_params)
                        loss = loss + fs[1]
                        instances_processed = instances_processed + target:size(1)
                        print('loss', loss/instances_processed)
                    end
                else
                    local pred = model:forward(inputs)
                    -- TODO: evaluate
                end
                collectgarbage()
            end
        end
    end
    exit()

    logger:add{['training error'] = loss }
    logger:style{['training error'] = '-'}
--    logger:plot()
end
