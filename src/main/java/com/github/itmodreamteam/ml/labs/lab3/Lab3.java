package com.github.itmodreamteam.ml.labs.lab3;

import com.github.itmodreamteam.ml.classification.*;
import com.github.itmodreamteam.ml.classification.bayes.BinaryNaiveBayesClassifier;
import com.github.itmodreamteam.ml.classification.bayes.BinaryNaiveBayesClassifierFactory;
import com.github.itmodreamteam.ml.metric.Metric;
import com.github.itmodreamteam.ml.validation.ClassifierCrossValidator;
import com.github.itmodreamteam.ml.validation.Samples;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.filefilter.RegexFileFilter;
import org.apache.commons.io.filefilter.TrueFileFilter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.nio.charset.Charset;
import java.util.*;

import static java.util.stream.Collectors.toList;

public class Lab3 {
    private static final Logger LOG = LoggerFactory.getLogger(Lab3.class);

    public static void main(final String... args) throws Exception {
        Collection<File> messages = FileUtils.listFiles(new File("."), new RegexFileFilter(".*legit.*|.*spmsg.*"), TrueFileFilter.INSTANCE);
        List<Boolean> spamOrNot = new ArrayList<>();
        List<List<Integer>> words = new ArrayList<>();
        List<List<Integer>> subjects = new ArrayList<>();
        List<List<Integer>> bodies = new ArrayList<>();
        List<Message> msgs = new ArrayList<>();
        for (File message : messages) {
            boolean spam = isSpamMessage(message);
            List<String> lines = FileUtils.readLines(message, Charset.defaultCharset());
            String subject = lines.get(0).substring("Subject: ".length());
            String body = lines.get(2);
            List<Integer> subjectCodes = getCodes(subject);
            List<Integer> bodyCodes = getCodes(body);
            List<Integer> all = new ArrayList<>();
            all.addAll(subjectCodes);
            all.addAll(bodyCodes);
            spamOrNot.add(spam);
            words.add(all);
            subjects.add(subjectCodes);
            bodies.add(bodyCodes);
            msgs.add(new Message(subjectCodes, bodyCodes));
        }

        LinkedHashSet<Boolean> classes = new LinkedHashSet<>();
        classes.add(true);
        classes.add(false);

        for (double threshold = -128; threshold <= 32; threshold += 10) {
            ClassifierFactory<List<Integer>, Boolean> factory = new BinaryNaiveBayesClassifierFactory<>(0.00001, threshold);
            ClassifierCrossValidator<List<Integer>, Boolean> validator = new ClassifierCrossValidator<>(factory, classes);
            Metric<Boolean> metric = validator.validate(Samples.of(words, spamOrNot), 10);
            LOG.info("threshold: {}, metric = {}", threshold, metric);
//            LOG.info("threshold: {}, f1 measure(spam) = {}", threshold, metric.f1measure(true));
//            LOG.info("threshold: {}, f1 measure(legit) = {}", threshold, metric.f1measure(false));
            LOG.info("threshold: {}, recall(spam) = {}", threshold, metric.recall(true));
            LOG.info("threshold: {}, recall(legit) = {}", threshold, metric.f1measure(false));
        }
        double alpha = 0.001;
        double threshold = 2;
        Classifier<Message, Boolean> allWords = GenericClassifier.wrap(new BinaryNaiveBayesClassifier<>(words, spamOrNot, alpha, threshold), Message::getAllWords);
        Classifier<Message, Boolean> onlySubject = GenericClassifier.wrap(new BinaryNaiveBayesClassifier<>(subjects, spamOrNot, alpha, threshold), Message::getSubject);
        Classifier<Message, Boolean> onlyBody = GenericClassifier.wrap(new BinaryNaiveBayesClassifier<>(bodies, spamOrNot, alpha, threshold), Message::getBody);

        List<Classifier<Message, Boolean>> classifiers = Arrays.asList(allWords, onlyBody, onlySubject);
        CompositeClassifierFactory<Message, Boolean> all = new CompositeClassifierFactory<>(classifiers, new AllQuorum());
        CompositeClassifierFactory<Message, Boolean> classic = new CompositeClassifierFactory<>(classifiers, new ClassicQuorum());

        ClassifierCrossValidator<Message, Boolean> validator1 = new ClassifierCrossValidator<>(all, classes);
        ClassifierCrossValidator<Message, Boolean> validator2 = new ClassifierCrossValidator<>(classic, classes);
        Metric<Boolean> metric1 = validator1.validate(Samples.of(msgs, spamOrNot), 10);
        Metric<Boolean> metric2 = validator2.validate(Samples.of(msgs, spamOrNot), 10);
        List<Metric<Boolean>> metrics = Arrays.asList(metric1, metric2);
        for (Metric<Boolean> metric : metrics) {
            LOG.info("metric: {}", metric1);
            LOG.info("metric f1 measure(spam): {}", metric.f1measure(true));
            LOG.info("metric f1 measure(legit): {}", metric.f1measure(false));
            LOG.info("metric recall(spam): {}", metric.recall(true));
            LOG.info("metric recall(legit): {}", metric.recall(false));
        }
    }

    private static class AllQuorum implements CompositeClassifier.Quorum<Boolean> {
        @Override
        public Boolean decide(List<Boolean> answers) {
            return answers.stream().allMatch(answer -> answer);
        }
    }

    private static class ClassicQuorum implements CompositeClassifier.Quorum<Boolean> {
        @Override
        public Boolean decide(List<Boolean> answers) {
            int spam = (int) answers.stream().filter(answer -> answer).count();
            int legit = answers.size() - spam;
            if (spam > legit) {
                return true;
            } else {
                return false;
            }
        }
    }

    private static List<Integer> getCodes(String line) {
        return Arrays.stream(line.split(" "))
                .map(String::trim)
                .filter(str -> !str.isEmpty())
                .mapToInt(Integer::parseInt)
                .boxed()
                .collect(toList());
    }

    private static boolean isSpamMessage(File message) {
        return message.getName().contains("spmsg");
    }
}
