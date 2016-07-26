--
-- Created by IntelliJ IDEA.
-- User: Ethan
-- Date: 7/19/16
-- Time: 11:17 PM
-- To change this template use File | Settings | File Templates.
--
require 'torch'
require 'csvigo'

local sets = {'train', 'test'}
for i = 1, #sets do
    local set = sets[i]
    for batchDir in paths.iterdirs(set) do
        local batchPath = paths.concat(set, batchDir)
        for csvFile in paths.iterfiles(batchPath) do
            if csvFile:find(".csv") then
                local filePath = paths.concat(batchPath, csvFile)
                local csvTable = csvigo.load(filePath, nil, 'raw')
                local tensor = torch.Tensor(csvTable)
                local filename = filePath:gsub('.csv', '.dat')
                torch.save(filename, tensor)
            end
        end
    end
end
