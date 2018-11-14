package com.github.itmodreamteam.ml.utils.collections;

import cern.colt.list.IntArrayList;
import com.github.itmodreamteam.ml.utils.collections.impl.colt.ColtIntList;

import java.util.Collection;

public class Lists {
    public static IntList of(int... elements) {
        return new ColtIntList(new IntArrayList(elements));
    }

    public static IntList of(Collection<Integer> elements) {
        int[] ints = elements.stream()
                .mapToInt(e -> e)
                .toArray();
        return new ColtIntList(new IntArrayList(ints));
    }
}
