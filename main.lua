--
-- Created by IntelliJ IDEA.
-- User: Ethan
-- Date: 7/9/16
-- Time: 11:41 AM
-- To change this template use File | Settings | File Templates.
--
require 'torch'
require 'rnn'

local batch_size = 2
local seq_len_in = 3
local seq_len_out = 4
local embed_dim = 5
local output_dim = 6
local num_layers = 2

local x = torch.ones(seq_len_in, batch_size, embed_dim)
print(x)
local y = torch.ones(seq_len_out, batch_size, output_dim)
local enc = nn.Sequential()
for _ = 1, num_layers - 1 do
    enc:add(nn.SeqGRU(embed_dim, embed_dim))
end
enc:add(nn.SeqGRU(embed_dim, output_dim))

local h = enc:forward(x) -- seq_len_in, batch_size, output_dim
print(h)
