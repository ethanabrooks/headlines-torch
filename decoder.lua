require 'rnn'
require 'nngraph'

function buildDecoder(train, hiddenSize, inSeqLen, outSeqLen, weightedModules)
    -- weighted modules
    local embedding, deepGRU, outLayer = unpack(weightedModules)

    --- build zipRepeating: {h, y} -> {{h, y1}, {h, y2}, ...}
    local zipRepeating = nn.ConcatTable()
    for i = 1, outSeqLen do
        local parallel = nn.ParallelTable()
        parallel:add(nn.Identity())   -- repeating
        parallel:add(nn.Select(1, i)) -- nonrepeating
        zipRepeating:add(parallel)
    end

    --- from now on, y refers to yi and s refers to si

    --- build start: {h, y} -> {h, y, s}
    local h, y = nn.Identity()(), nn.Identity()()
    local start = nn.gModule({ h, y }, { h, y, nn.Select(1, inSeqLen)(h) })
                                            -- last state of encoder is
                                            -- first state of decoder
    --TODO: maybe not the best option

    --- build merge for nn.Recurrent(): {{h, y}, {s, y_pred}} -> {h, y, s}
    local merge
    if train then
        -- feed correct answer at each time step
        local table, s   = nn.Identity()(), nn.SelectTable(1)()
        local h          = nn.SelectTable(1)(table)
        local y          = nn.SelectTable(2)(table)
        merge            = nn.gModule({ table, s }, { h, y, s })
    else
        -- feed previous prediction back in
        local h, table  = nn.SelectTable(1)(), nn.Identity()()
        local s         = nn.SelectTable(1)(table)
        local y_pred    = nn.SelectTable(2)(table)
        local y         = nn.ArgMax(2, 2)(y_pred)
        merge           = nn.gModule({ h, table }, { h, y, s })
    end

    --- build transfer: {h, y, s_tm1} -> {s, y_pred}
    local make2d        = nn.Reshape(-1, hiddenSize, false) -- for weighting across timesteps and batches
    local restoreShape = nn.Reshape(inSeqLen, -1, hiddenSize, false)

    local combine = nn.Sequential()        -- combines s_tm1 with embed(y)
    combine:add(nn.CAddTable())            -- combine though addition
    combine:add(nn.Replicate(inSeqLen, 1)) -- broadcast
    combine:add(make2d)                    -- for alignment with h

    local y, s_tm1 = embedding(), nn.Identity()()  -- both batchSize, hiddenSize
    local h, y_s   = make2d(), combine{ y, s_tm1 } -- both inSeqLen * batchSize, hiddenSize
    local align    = nn.Sequential()

    align:add(nn.CosineDistance())         -- compare   (result: inSeqLen * batchSize)
    align:add(nn.SoftMax())                -- normalize (result: inSeqLen * batchSize)
    align:add(nn.Replicate(hiddenSize, 2)) -- broadcast (result: inSeqLen * batchSize, hiddenSize)
    local attention = align{ h, y_s }

    -- apply attention and pass through deep GRU
    local process = nn.Sequential()

    -- dot product
    process:add(nn.CMulTable()) -- inSeqLen * batchSize, hiddenSize
    process:add(restoreShape)   -- inSeqLen, batchSize, hiddenSize
    process:add(nn.Sum(1))      -- batchSize, hiddenSize

    process:add(deepGRU)                            -- batchSize, hiddenSize
    local s = process{ h, attention }               -- batchSize, hiddenSize
    local transfer = nn.gModule({ h, y, s_tm1 },    -- sizes specified above
                                { s,                -- batchSize, hiddenSize,
                                  outLayer(s) })    -- batchSize, nClasses }

    --- build module for nn.Sequencer
    local module = nn.Sequential()
    module:add(nn.Recurrent( -- Recurrent layer, the main workhorse of the decoder
    --  { (on step 1, start ->) input, feedback } -> merge -> transfer
        start,
        nn.Identity(), -- input    {h, y} -> {h, y}
        nn.Identity(), -- feedback {s, y_pred} -> {s, y_pred}
        transfer,      --          {h, y, s_tm1} -> {s, y_pred}
        outSeqLen,     -- rho
        merge          --          {{h, y}, {s, y_pred}} -> {h, y, s}
    ))
    module:add(nn.SelectTable(2)) -- {s, y_pred} -> y_pred

    local decoder = nn.Sequential()
    decoder:add(zipRepeating)
    decoder:add(nn.Sequencer(module, inSeqLen))
    return decoder
end
return buildDecoder
