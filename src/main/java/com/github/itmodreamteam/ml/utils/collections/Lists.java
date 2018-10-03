package com.github.itmodreamteam.ml.utils.collections;

import cern.colt.list.IntArrayList;
import com.github.itmodreamteam.ml.utils.collections.impl.colt.ColtIntList;

public class Lists {
    public static IntList of(int... elements) {
        return new ColtIntList(new IntArrayList(elements));
    }
}
