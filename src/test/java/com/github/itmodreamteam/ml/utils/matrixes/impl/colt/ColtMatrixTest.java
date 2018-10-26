package com.github.itmodreamteam.ml.utils.matrixes.impl.colt;

import com.github.itmodreamteam.ml.utils.matrixes.AbstractMatrixTest;

public class ColtMatrixTest extends AbstractMatrixTest {
    public ColtMatrixTest() {
        super(new ColtVectorFactory(), new ColtMatrixFactory());
    }
}