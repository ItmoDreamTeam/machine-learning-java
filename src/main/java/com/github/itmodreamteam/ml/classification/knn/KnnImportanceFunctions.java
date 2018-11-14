package com.github.itmodreamteam.ml.classification.knn;

public enum KnnImportanceFunctions implements KnnImportanceFunction {
    EPANCHIKOV {
        @Override
        public double importance(double distance) {
            return 3.0 / 4.0 * (1 - distance * distance);
        }
    },
    TRIANGLE {
        @Override
        public double importance(double distance) {
            return 1 - distance;
        }
    },
    BIWEIGHT {
        @Override
        public double importance(double distance) {
            return 15.0 / 16.0 * Math.pow(1 - distance * distance, 2);
        }
    },
    TRIWEIGHT {
        @Override
        public double importance(double distance) {
            return 35.0 / 32.0 * Math.pow(1 - distance * distance, 3);
        }
    },
    TRICUBE {
        @Override
        public double importance(double distance) {
            return 70.0 / 81.0 * (Math.pow(1 - Math.pow(distance, 3), 3));
        }
    },
    COS {
        @Override
        public double importance(double distance) {
            return Math.PI / 4 * Math.cos(Math.PI / 2 * distance);
        }
    }
}
