require 'rnn'
require 'nngraph'
require 'ZipRepeating'


function decoder(train, inSeqLen, outSeqLen)
    -- hyper-parameters
    local batchSize = 3
    local hiddenSize = 2
    local nIndex = 1000
    local nClasses = 6
    local depth = 3

    --- build start: {h, y} -> {h, y, s}
    local h, y = nn.Identity()(), nn.Identity()()
    local start = nn.gModule({ h, y }, { h, y, nn.Select(1, inSeqLen)(h) })

    --- build merge for r_train: {{h, y}, {s, y_pred}} -> {h, y, s}
    local table, s = nn.Identity()(), nn.SelectTable(1)()
    local h = nn.SelectTable(1)(table)
    local y = nn.SelectTable(2)(table)
    local merge = nn.gModule({ table, s }, { h, y, s })

    --- build transfer: {h, s} -> t
    local make2d  = nn.Reshape(batchSize * inSeqLen, hiddenSize, false) -- for weighting across timesteps and batches
    local combine = nn.Sequential() -- combines s_tm1 with embed(y)
    combine:add(nn.CAddTable())
    combine:add(nn.Replicate(inSeqLen, 1))
    combine:add(make2d)

    local y, s_tm1 = nn.LookupTable(nIndex, hiddenSize)(), nn.Identity()() -- both batchSize, hiddenSize
    local h, y_s   = make2d(), combine{ y, s_tm1 } -- both inSeqLen * batchSize, hiddenSize
    local attention = nn.Replicate(hiddenSize, 2) -- broadcast
                                  (nn.CosineDistance(){ h, y_s }) -- inSeqLen * batchSize, hiddenSize

    -- apply attention and pass through deep GRU
    local process = nn.Sequential()
    -- dot produce
    process:add(nn.CMulTable())
    process:add(nn.Reshape(inSeqLen, batchSize, hiddenSize, false))
    process:add(nn.Sum(1))
    -- deep GRU
    for _ = 1, depth do
        process:add(nn.GRU(hiddenSize, hiddenSize))
    end

    local s = process{ h, attention }
    local yPred = nn.Linear(hiddenSize, nClasses)(s) -- distribution of predictions
    local transfer = nn.gModule({ h, y, s_tm1 },
        { s, yPred }) -- batchSize, hiddenSize

    --- build Recurrent layer, the main workhorse of the decoder
    local rTrain = nn.Recurrent(
        start, -- start
        nn.Identity(), -- input    {h, y} -> {h, y}
        nn.Identity(), -- feedback {s, y_pred} -> {s, y_pred}
        transfer,      --          {h, y, s} -> {s, y_pred}
        outSeqLen,     -- rho
        merge          --          {{h, y}, {s, y_pred}} -> {h, y, s}
    )

    --- build merge for r_test: {{h, y}, {s, y_pred}} -> {h, y, s}
    local h, table = nn.SelectTable(1)(), nn.Identity()()
    local s = nn.SelectTable(1)(table)
    local y_pred = nn.SelectTable(2)(table)
    local y = nn.ArgMax(2, 2)(y_pred)
    local testMerge = nn.gModule({ h, table }, { h, y, s })

    local seqModule = nn.Sequential()
    seqModule:add(rTrain)
    seqModule:add(nn.SelectTable(2))

    --- build rTest from rTrain
    local rTest = rTrain:sharedClone()
    rTest.mergeModule = testMerge
    rTest:buildRecurrentModule()
    -- TODO sharing. ensure that rTest's parameters come from rTrain

    if not train then
        seqModule.modules[1] = rTest
    end

    return nn.Sequencer(seqModule, inSeqLen)
end
return decoder
