package edu.stanford.nlp.semeval2015.pit;

import static edu.stanford.nlp.semeval2015.pit.ParaphraseProblem.readDefinite;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.Ordering;
import com.google.common.collect.Sets;

import edu.stanford.nlp.classify.Classifier;
import edu.stanford.nlp.classify.LinearClassifierFactory;
import edu.stanford.nlp.classify.RVFDataset;
import edu.stanford.nlp.kbp.entitylinking.scripts.JaroWinkler;
import edu.stanford.nlp.semeval2015.pit.Phrase.WordPair;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.stats.OpenAddressCounter;

class Evaluation {
  
  public enum ErrorClass {
    TruePositive, FalsePositive, TrueNegative, FalseNegative;
    public boolean isTrue() {
      return this == TruePositive || this == TrueNegative;
    }
    
    public boolean isFalse() {
      return this == FalsePositive || this == FalseNegative;
    }
  }

  /***
   * Adapted from {@link edu.stanford.nlp.stock.Scorer.F1Score}
   *
   */
  public static class Performance {

    private double truePositives, falsePositives, trueNegatives, falseNegatives;

    public double getPrecision() {
      return truePositives / (truePositives + falsePositives);
    }

    public double getRecall() {
      return truePositives / (truePositives + falseNegatives);
    }

    public double getF1() {
      double p = getPrecision();
      double r = getRecall();
      return 2 * p * r / (p + r);
    }

    public double getAccuracy() {
      return (truePositives + trueNegatives)
          / (truePositives + falsePositives + trueNegatives + falseNegatives);
    }

    public void printF1Score() {
      System.out.printf("F1: %.3f\tP: %.3f\tR: %.3f\tAccuracy: %.3f\n", getF1(),
          getPrecision(), getRecall(), getAccuracy());
    }

    /***
     * Updates performance stats for binary prediction
     * 
     * @param predicted
     *          label
     * @param actual
     *          label
     */
    public ErrorClass update(boolean predict, boolean actual) {
      if (predict)
        if (actual) {
          truePositives++;
          return ErrorClass.TruePositive;
        } else {
          falsePositives++;
          return ErrorClass.FalsePositive;
        }
      else if (actual) {
        falseNegatives++;
        return ErrorClass.FalseNegative;
      } else {
        trueNegatives++;
        return ErrorClass.TrueNegative;
      }
    }

  }


  static void evaluateSocher() throws IOException {
    List<String> strlabels = Files.readAllLines(Paths.get(Params.socherOutput));
    List<Boolean> labels = Lists.transform(strlabels, s -> "1".equals(s));
    List<ParaphraseProblem> problems = readDefinite(Params.testingData);
    Performance performance = new Performance();
    IntStream.range(0, labels.size()).forEach(i -> {
      boolean predict = labels.get(i);
      boolean actual = problems.get(i).label.isTrue();
      ErrorClass e = performance.update(predict, actual);
      if (!e.isTrue()) {
        System.out.println(problems.get(i));
        System.out.println();
      }
    });
    performance.printF1Score();
  }

  static void evaluate() throws IOException, ClassNotFoundException {

    RVFDataset<Boolean, Integer> data = new RVFDataset<>();
    for(ParaphraseProblem p:readDefinite(Params.trainingData)){
      data.add(Features.getFeatureDatum(p));
    }

    final Classifier<Boolean, Integer> c =

//    new LogisticClassifierFactory<Boolean, Integer>().trainClassifier(data);
     new LinearClassifierFactory<Boolean, Integer>().trainClassifier(data);
//     LibSVMClassifier.trainLinearSVM(data, 0.1);
//    LibSVMClassifier.trainPolynomialSVM(data);
//    LibSVMClassifier.trainRBFSVM(data, 0.8, 1.0, false, true);

    // System.out.println(LibSVMClassifier.CVAccuracyLinear(data,1));
//     System.out.println(LibSVMClassifier.CVAccuracyRBF(data,0.5,1));

    Performance performance = new Performance();
    Counter<String> errors = new OpenAddressCounter<String>();
    List<ParaphraseProblem> testingProblems = ParaphraseProblem.readAll(Params.testingData);

//    Collections.shuffle(testingProblems);
    for (ParaphraseProblem p : testingProblems) {

      boolean predict = c.classOf(Features.getFeatureDatum(p));
      boolean actual = p.label.isTrue();
      ErrorClass error = performance.update(predict, actual);
      errors.incrementCount(error.name());
      if (error == ErrorClass.FalseNegative && !p.label.isDebatable()) {
        // IOs.enableConsole();
        
//        if(p.s2.surface.equals("That amber alert was getting annoying")
//            ){
//         System.out.println(p);
//         p.s1.getDependencies().prettyPrint();
//         System.out.println();
//        }
        
         // IOs.silenceConsole();
        // errors.incrementCount(p.label.toString());
        System.out.println(p);
        System.out.println(p.s1.getAllPhrases(p.topicName, 3));
//        System.out.println();
      }
    }
    System.out.println(errors);
    performance.printF1Score();
  }
  
  static void problemStats() throws IOException {

    readDefinite(Params.trainingData).stream()
    .forEach(p->{
      System.out.println(Features.getBaselineNgramFeatures(p));
    }
    );
    System.exit(0);
    
    List<ParaphraseProblem> problems = readDefinite(Params.trainingData);

    Collections.shuffle(problems);
    problems
        .stream()
        .filter(p -> p.label.isTrue())
        .forEach(p -> {

          Set<String> tokens1 = Sets.newHashSet(p.s1.getShingles(3));
          Set<String> tokens2 = Sets.newHashSet(p.s2.getShingles(3));

          Set<String> intersection = Sets.intersection(tokens1, tokens2);

          Set<String> uniq1 = Sets.filter(tokens1, s -> !intersection.contains(s));
          Set<String> uniq2 = Sets.filter(tokens2, s -> !intersection.contains(s));

          // p.s1.getEntitySegmentedPhrases(3)
            Iterable<Phrase> sequences1 = uniq1.stream()
                .map(s -> new Phrase(s, Phrase.vectorize(s)))
                .filter(w -> w.isNonStop() && w.hasKnownWordVector()) 
                .collect(Collectors.toList());
            Iterable<Phrase> sequences2 = uniq2.stream()
                .map(s -> new Phrase(s, Phrase.vectorize(s)))
                .filter(w -> w.isNonStop() && w.hasKnownWordVector())
                .collect(Collectors.toList());

            List<WordPair> pairs = Lists.newArrayList();

            for (Phrase w1 : sequences1) {
              for (Phrase w2 : sequences2) {
                // ignore same words
                if (!w1.surface.equalsIgnoreCase(w2.surface)) {
                  pairs.add(new WordPair(w1, w2));
                }
              }
            }
            pairs = Ordering.natural().greatestOf(
                Iterables.filter(pairs, x -> x.score > 0.3), 10);
            if (pairs.size() > 0) {
              System.out.println(pairs);
              System.out.println(p);
              // System.out.println(p.s1.getNonEntityPhrases(3));
              // System.out.println(p.s1.getChunks());
              // System.out.println(p.s1.getEntities());
              System.out.println();
            }
          });

    // IntStream lengths = problems.stream().flatMapToInt(
    // p -> IntStream.of(p.s1.words.size(), p.s2.words.size()));
    //
    // int[] lens = lengths.toArray();
    // Counter<Integer> c = Counters.asCounter(Ints.asList(lens));
    // System.out.println(c);
    // System.out.println("distribution: " + Counters.toVerticalString(c));
    // System.out.println("max: " + Counters.max(c));
    // System.out.println("min: " + Counters.min(c));
    // System.out.println("mean: " + Counters.mean(c));
    System.exit(0);
  }
  
  static void topicCoverage() throws IOException{
    int noTopic = 0;
    int total = 0;
    JaroWinkler.similarity("", "");
    for(ParaphraseProblem p:ParaphraseProblem.readAllProblems()){
      total++;
      boolean s1contains = p.s1.surface.toLowerCase().contains(p.topicName.toLowerCase());
      boolean s2contains = p.s2.surface.toLowerCase().contains(p.topicName.toLowerCase());
      if(!s1contains || !s2contains){
        noTopic++;
//        System.out.println(p);
//        System.out.println();
      }
    }
    System.out.printf("Total:%d;No topic string:%d\n",total, noTopic);
  }

}
