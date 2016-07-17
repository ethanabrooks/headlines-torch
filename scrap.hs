#!/usr/local/bin/runhaskell


main = do
    print $ zipRepeating 1 [2..4]

zipRepeating :: a -> [b] -> [(a, b)]
zipRepeating repeating list =
    map (\ elt -> (repeating, elt)) list

