package com.github.itmodreamteam.ml.genetic;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.IntStream;

import static java.util.stream.Collectors.toList;

public class GeneticAlgorithm<T extends Individual> {
    private static final Logger LOGGER = LoggerFactory.getLogger(GeneticAlgorithm.class);
    private static final ThreadLocalRandom RANDOM = ThreadLocalRandom.current();
    private static final int SCALE = 1_000_000;
    private final int numberOfGenerations;
    private final int initialGenerationSize;
    private final int selectionSize;
    private final int numberOfGenes;
    private final Comparator<T> comparator;
    private final boolean killParents;
    private final int mutationProbability;
    private final IndividualFactory<T> factory;

    public GeneticAlgorithm(int numberOfGenerations, int initialGenerationSize, int selectionSize, int numberOfGenes, Comparator<T> comparator, boolean killParents, int mutationProbability, IndividualFactory<T> factory) {
        this.numberOfGenerations = numberOfGenerations;
        this.initialGenerationSize = initialGenerationSize;
        this.selectionSize = selectionSize;
        this.numberOfGenes = numberOfGenes;
        this.comparator = comparator;
        this.killParents = killParents;
        this.mutationProbability = mutationProbability;
        this.factory = factory;
    }

    public T make() {
        List<T> generation = createInitialGeneration();
        for (int generationNumber = 0; generationNumber < numberOfGenerations; ++generationNumber) {
            generation = runGeneration(generation);
            LOGGER.info("generation: {}", generationNumber, generation.size());

        }
        return chooseBest(generation, 1).get(0);
    }

    private List<T> runGeneration(List<T> generation) {
        List<T> selection = chooseBest(generation, selectionSize);
        List<T> children = new ArrayList<>();
        for (T first : selection) {
            for (T second : selection) {
                T child = makeChild(first, second);
                children.add(child);
            }
        }

        List<T> nextGeneration = new ArrayList<>();

        if (!killParents) {
            nextGeneration.addAll(selection);
        }
        nextGeneration.addAll(children);
        nextGeneration.addAll(createInitialGeneration());
        return nextGeneration;
    }

    private List<T> chooseBest(List<T> generation, int selectionSize) {
        Map<T, Integer> score = new HashMap<>();
        for (T first : generation) {
            for (T second : generation) {
                if (comparator.compare(first, second) > 0) {
                    score.merge(first, 1, (i1, i2) -> i1 + i2);
                } else {
                    score.merge(second, 1, (i1, i2) -> i1 + i2);
                }
            }
        }

        return score.entrySet()
                .stream()
                .sorted((e1, e2) -> Integer.compare(e2.getValue(), e1.getValue()))
                .limit(selectionSize)
                .map(Map.Entry::getKey)
                .collect(toList());
    }

    private T makeChild(T first, T second) {
        int crossLine = RANDOM.nextInt(numberOfGenes);
        double[] childGenes = new double[numberOfGenes];
        System.arraycopy(first.getGenes(), 0, childGenes, 0, crossLine);
        System.arraycopy(second.getGenes(), crossLine, childGenes, crossLine, numberOfGenes - crossLine);
        for (int genNumber = 0; genNumber < numberOfGenes; ++genNumber) {
            if (shouldMutate()) {
                childGenes[genNumber] = mutate(childGenes[genNumber]);
            }
        }

        return factory.create(childGenes);
    }

    private double mutate(double gene) {
        return (1 + RANDOM.nextDouble(-0.3, 0.3)) * gene;
    }

    private boolean shouldMutate() {
        return RANDOM.nextInt(100) < mutationProbability;
    }

    private List<T> createInitialGeneration() {
        return IntStream.range(0, initialGenerationSize)
                .mapToObj(i -> generateRandom())
                .collect(toList());
    }

    private T generateRandom() {
        double[] genes = new double[numberOfGenes];
        for (int genNumber = 0; genNumber < numberOfGenes; ++genNumber) {
            genes[genNumber] = RANDOM.nextDouble() * SCALE;
        }
        return factory.create(genes);
    }
}
