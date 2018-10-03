package com.github.itmodreamteam.ml.utils.collections.impl.colt;

import cern.colt.list.IntArrayList;
import com.github.itmodreamteam.ml.utils.collections.IntList;

import java.util.function.IntPredicate;

public class ColtIntList implements IntList {
    private final IntArrayList colt;

    public ColtIntList(IntArrayList colt) {
        this.colt = colt;
    }

    @Override
    public int size() {
        return colt.size();
    }

    @Override
    public int get(int index) {
        return colt.get(index);
    }

    @Override
    public int[] toArray() {
        return colt.elements();
    }

    @Override
    public IntList slice(IntPredicate index) {
        int[] elements = colt.elements();
        IntArrayList res = new IntArrayList();
        for (int i = 0; i < elements.length; ++i) {
            if (index.test(i)) {
                res.add(elements[i]);
            }
        }
        return new ColtIntList(res);
    }
}
