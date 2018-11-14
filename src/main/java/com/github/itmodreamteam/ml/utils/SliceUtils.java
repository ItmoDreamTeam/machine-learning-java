package com.github.itmodreamteam.ml.utils;

import java.util.ArrayList;
import java.util.List;
import java.util.function.IntPredicate;

public class SliceUtils {
    public static <T> List<T> slice(List<T> data, IntPredicate predicate) {
        List<T> result = new ArrayList<>();
        for (int i = 0; i < data.size(); ++i) {
            if (predicate.test(i)) {
                result.add(data.get(i));
            }
        }
        return result;
    }
}
