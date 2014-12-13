package edu.stanford.nlp.semeval2015.pit;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.neural.rnn.RNNCoreAnnotations;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.sentiment.SentimentCoreAnnotations;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.util.CoreMap;

/**
 * Problem model for Semeval 2015 task 1 Paraphrase in Twitter
 * 
 * http://alt.qcri.org/semeval2015/task1/
 *
 */
public class ParaphraseProblem implements Iterable<Sentence> {

  public final int num;
  public final String topicName;
  public final Sentence s1;
  public final Sentence s2;
  public final Label label;

  private static final int DATE = 0;
  private static final int TOPIC = 1;
  private static final int Sentence1 = 2;
  private static final int Sentence2 = 3;
  private static final int LABEL = 4;
  private static final int S1TAG = 5;
  private static final int S2TAG = 6;

  public ParaphraseProblem(String line) {
    String[] cols = line.split("\t");
    num = Integer.parseInt(cols[DATE]);
    topicName = cols[TOPIC];
    s1 = new Sentence(cols[Sentence1], cols[S1TAG]);
    s2 = new Sentence(cols[Sentence2], cols[S2TAG]);
    label = new Label(cols[LABEL]);
  }

  /***
   * 
   * @return shingles up to trigram
   */
  public Stream<String> getShingleStream() {
    return getShingleStream(3);
  }

  /***
   * 
   * @param maxSize
   *          indicates the largest span of interest
   * @return the shingles of the sentence pair size 1 all the way up to maxSize
   */
  public Stream<String> getShingleStream(int maxSize) {
    return Stream.concat(s1.getShingles(maxSize).stream(), s2.getShingles(maxSize)
        .stream());
  }

  @Override
  public Iterator<Sentence> iterator() {
    return Arrays.asList(s1, s2).iterator();
  }

  public String toString() {
    return topicName + " - " + label + "\n" + s1 + "\n" + s2;
  }
  
  public boolean isValidTrainingData(){
    return !label.isDebatable() && !s1.surface.equals(s2.surface);
  }
  
  /***
   * Static utility methods
   */

  /***
   * Reads in a stream of problems from the source file
   * 
   * @param filename
   * @return
   * @throws IOException
   */
  static Stream<ParaphraseProblem> read(String filename) throws IOException {
    return Files.lines(Paths.get(filename)).map(ParaphraseProblem::new);
  }
  
  /**
   * Reads in the list of problems
   * @param loc
   * @return
   * @throws IOException
   */
  static List<ParaphraseProblem> readDefinite(String loc) throws IOException{
    return read(loc)
//        .parallel()
        .filter(ParaphraseProblem::isValidTrainingData)
        .collect(Collectors.toList());
  }
  
  /**
   * Reads in the list of problems
   * @param loc
   * @return
   * @throws IOException
   */
  static List<ParaphraseProblem> readAll(String loc) throws IOException{
    return read(loc)
        .parallel()
        .collect(Collectors.toList());
  }
  
  static List<ParaphraseProblem> readAllProblems() throws IOException{
    return Stream.concat(read(Params.trainingData), read(Params.testingData))
        .parallel()
        .unordered().collect(Collectors.toList());
  }

  public static void main(String[] args) throws Exception {
//    IOs.cacheAllAnnotation();
    Evaluation.evaluate();
  }

}
