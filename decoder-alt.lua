require 'rnn'
require 'nngraph'


function buildDecoder(train, hiddenSize, inSeqLen, outSeqLen, weightedModules)
    -- weighted modules
    local embed, deepGRU, outLayer = unpack(weightedModules)

    --- build zipRepeating: {h, y} -> {{h, y1}, {h, y2}, ...}
    ---------------------------------------------------------
    --- h:      [batchSize, inSeqLen, hiddenSize]
    --- y:      [batchSize, outSeqLen, nClasses]
    --- output: {{[batchSize, inSeqLen, hiddenSize], [batchSize, nClasses]}}
    local y            = nn.SplitTable(2)()
    local h            = nn.Identity()()
    local output       = nn.ZipTableOneToMany(){h, y}
    local zipRepeating = nn.gModule({h, y}, {output})

    --- from now on, y refers to yi and s refers to si

    local input = nn.Identity()()          -- {y, h}
    local h     = nn.SelectTable(1)(input) -- [batchSize, inSeqLen, hiddenSize]
    local y     = nn.SelectTable(2)(input) -- [batchSize, nClasses]
    local s     = nn.Identity()()          -- [batchSize, hiddenSize]

    local broadcast = nn.Replicate(inSeqLen, 2)(s)                 -- [batchSize, inSeqLen, hiddenSize]
    local h2d       = nn.Reshape(-1, hiddenSize, false)(h)         -- [batchSize * inSeqLen, hiddenSize]
    local s2d       = nn.Reshape(-1, hiddenSize, false)(broadcast) -- [batchSize * inSeqLen, hiddenSize]

    --- {s, h} -> attention over h
    ------------------------------
    --- s:      [inSeqLen * batchSize, hiddenSize]
    --- h:      [inSeqLen * batchSize, hiddenSize]
    --- output: [inSeqLen * batchSize, hiddenSize]
    local align = nn.Sequential()
    :add(nn.CosineDistance())              -- compare                   [inSeqLen * batchSize]
    :add(nn.Reshape(-1, inSeqLen, false))  -- expose inSeqLen dimension [batchSize, inSeqLen]
    :add(nn.SoftMax())                     -- softmax over inSeqLen dim [inSeqLen, batchSize]
    :add(nn.Reshape(-1, false))            -- ravel                     [batchSize * inSeqLen]
    :add(nn.Replicate(hiddenSize, 2))      -- broadcast                 [batchSize * inSeqLen, hiddenSize]
    local attention = align{s2d, h2d}      --                           [batchSize * inSeqLen, hiddenSize]

    --- {attention, h} -> attention(h)
    ----------------------------------
    --- attention: [inSeqLen * batchSize, hiddenSize]
    --- h:         [batchSize, hiddenSize]
    --- output:    [batchSize, hiddenSize]
    local apply = nn.Sequential()
    -- dot product
    :add(nn.CMulTable())                              -- [batchSize * inSeqLen, hiddenSize]
    :add(nn.Reshape(-1, inSeqLen, hiddenSize, false)) -- [batchSize, inSeqLen, hiddenSize]
    :add(nn.Sum(2))                                   -- [batchSize, hiddenSize]
    local weighted_h = apply{attention, h2d}          -- [batchSize, hiddenSize]

    --- {weighted_h, y, s} -> concat{weighted_h, y, s}
    --------------------------------------------------
    --- weighted_h: [batchSize, hiddenSize]
    --- y:          [batchSize, hiddenSize]
    --- s:          [batchSize, hiddenSize]
    local concat = nn.Concat(2)
    :add(nn.SelectTable(1))
    :add(nn.SelectTable(2))
    :add(nn.SelectTable(3))
    local gruInput = concat{weighted_h, y, s} -- [batchSize, 2 * hiddenSize + nClasses]
    local output   = deepGRU(gruInput)        -- [batchSize, hiddenSize]
    local rm = nn.gModule({ input, s }, { output })

    local sequencer = nn.Sequencer(
        nn.Sequential()
        :add(nn.Recurrence(rm, hiddenSize, 2, 1))
        :add(outLayer)
    )
    local decoder = nn.Sequential()
    :add(zipRepeating)
    :add(sequencer)
    return decoder
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

    local h = torch.range(1, inSeqLen * batchSize * hiddenSize)
    :resize(batchSize, inSeqLen, hiddenSize)
    local y = torch.range(1, outSeqLen * batchSize * nClasses)
    :resize(batchSize, outSeqLen, nClasses)
    print(y:size())
    print(h:size())

    local weighted = {nn.LookupTable(vocSize, hiddenSize), nn.GRU(hiddenSize * 2 + nClasses, hiddenSize), nn.Linear(hiddenSize, nClasses)}
    local model = buildDecoder(false, hiddenSize, inSeqLen, outSeqLen, weighted)
    local out = model:forward{ h, y }
    print(out)
    local criterion = nn.SequencerCriterion(nn.CrossEntropyCriterion())
    local tgts = torch.ones(batchSize, outSeqLen):split(1, 2)
    print(criterion:forward(out, tgts))
--    local out = model:forward{ x, y2 }
--    local gradOutput = model:backward({ x, y1 }, out)
--    model.model:updateParameters(1)
--    local out = model:forward{ x, y1 }
--    local out = testModel:forward{ x, y2 }
--    testModel:backward({ x, y2 }, out)
end
--test()
return buildDecoder

