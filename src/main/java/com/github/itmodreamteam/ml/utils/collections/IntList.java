package com.github.itmodreamteam.ml.utils.collections;

import java.util.function.IntPredicate;

public interface IntList {
    int size();

    int get(int index);

    int[] toArray();

    IntList slice(IntPredicate index);
}
