package edu.stanford.nlp.semeval2015.pit;

import static edu.stanford.nlp.math.ArrayMath.dotProduct;
import static edu.stanford.nlp.math.ArrayMath.floatArrayToDoubleArray;
import static edu.stanford.nlp.math.ArrayMath.norm;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Collection;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.lang3.StringUtils;
import org.apache.commons.lang3.mutable.MutableInt;
import org.apache.commons.lang3.text.WordUtils;
import org.tartarus.snowball.ext.PorterStemmer;

import com.google.common.collect.HashMultimap;
import com.google.common.collect.Multimap;
import com.google.common.collect.Sets;

import edu.stanford.cs.ra.util.IOUtils;
import edu.stanford.nlp.math.ArrayMath;

/**
 * Word representation for the twitter parse output given in the
 * training/testing data
 *
 */
class Phrase {

  static class WordPair implements Comparable<WordPair> {
    final Phrase w1;
    final Phrase w2;
    double[] v1;
    double[] v2;
    final boolean areSameWord;
    public double score;

    public WordPair(Phrase word1, Phrase word2) {
      this.w1 = word1;
      this.w2 = word2;
      this.v1 = w1.vectorize();
      areSameWord = w1.surface.equalsIgnoreCase(w2.surface);
      if(areSameWord){
        v2 = v1;
        score = 1;
      }else{
        this.v2 = w2.vectorize();
        score = Features.cosineSim(v1, v2);
      }
    }
    
    public boolean hasStopWord(){
      return w1.isStop() || w2.isStop();
    }

    @Override
    public int compareTo(WordPair other) {
      return Double.compare(score, other.score);
    }

    public String toString() {
      return String.format("%s ~ %s : %.2f", w1, w2, score);
    }

  }

  /**
   * string to vector representation mappings
   * multi word unit separated by space
   */
  private static Map<String, double[]> vectors;
  private static Multimap<String, String> lowercaseVectors;
  public static Set<String> stops;

  private static double[] zeroVector;
  static {
    try {
      stops = Sets.newHashSet(IOUtils.readLines(Phrase.class
          .getResourceAsStream("stops.txt")));

      // Cache word vectors in training data for faster experiments
      if (!new File(Params.cachedVectorsFile()).exists()) {
        System.out.println("Word vectors not cached, caching...");
        Stream<String> phraseStream = Stream.concat(
            ParaphraseProblem.read(Params.trainingData).flatMap(
                ParaphraseProblem::getShingleStream),
            ParaphraseProblem.read(Params.testingData).flatMap(
                ParaphraseProblem::getShingleStream));
        // Everything in lower-case to retrieve all matching words regardless of
        // cases
        Set<String> phraseSet = phraseStream.map(String::toLowerCase).collect(
            Collectors.toSet());
        Phrase.saveWordVectorSubset(phraseSet, Params.wordVectorFile,
            Params.cachedVectorsFile());
      }
      Phrase.loadWordVectors(Params.cachedVectorsFile());
    } catch (IOException e) {
      e.printStackTrace();
      System.exit(-1);
    }
  }

  public String surface;
  public String NETag;
  public String POSTag;
  public String chunkTag;
  public String eventTag;
  private double[] cachedVector = null;
  
  public Phrase(String surface){
    this(surface,vectorize(surface));
  }

  public Phrase(String taggedWord,boolean taggeed) {
    String[] parts = taggedWord.split("/");
    surface = parts[0];
    NETag = parts[1];
    POSTag = parts[2];
    chunkTag = parts[3];
    eventTag = parts[4];
  }

  Phrase(String surface, float[] vector) {
    this(surface, vector == null ? null : floatArrayToDoubleArray(vector));
  }
  
  Phrase(String surface, double[] vector) {
    this.surface = surface;
    cachedVector = vector;
  }

  /**
   * Finds the best vector representation for the current word
   * 
   * @return
   */
  public double[] vectorize() {
    if (cachedVector == null) {
      cachedVector = vectorize(surface);
    }
    return cachedVector;
  }
  
  /**
   * Whether this word is in the vocab of vectors
   * 
   * @return
   */
  public boolean hasKnownWordVector() {
    double[] vector = vectorize();
    return vector != null && vector != zeroVector;
  }

  public String toString() {
    return surface;
  }

  public boolean isStop() {
    return stops.contains(surface.toLowerCase());
  }
  
  public boolean isNonStop() {
    return !isStop();
  }

  public String toVectorStr() {
    return surface.replace(' ', '_') + " " + StringUtils.join(ArrayUtils.toObject(vectorize()), ' ');
  }

  /***
   * static utilities
   * 
   */
  public static boolean isStop(String word){
    return stops.contains(word.toLowerCase());
  }
  
  private static double[] parseVectors(String[] cols) {
    return Arrays.stream(cols, 1, cols.length)
        .mapToDouble(Double::parseDouble)
        .toArray();
  }

  /**
   * Retrieves a vector representation for a surface
   * 
   * @param word
   * @return
   */
  public static double[] vectorize(String word) {
    double[] ret = vectors.get(word);
    if (ret == null) {
      if(StringUtils.isAllLowerCase(word))
        return zeroVector;
      String lower = word.toLowerCase();
      ret = vectors.get(lower);
      if (ret == null) {
        if (stops.contains(lower))
          return zeroVector;
        Collection<String> alternativeNames = lowercaseVectors.get(lower);
        if (alternativeNames.isEmpty())
          return zeroVector;

        // First check for capitalized word
        String capitalized = WordUtils.capitalize(lower);
        if (word.equals(lower))
          return vectors.get(capitalized);
        return vectors.get(alternativeNames.iterator().next());
      }
    }
    return ret;
  }
  
  public static double cosSim(String w1,String w2){
    return Features.cosineSim(vectorize(w1), vectorize(w2));
  }
  
  public static double l2Sim(String w1,String w2){
    return Features.l2Sim(vectorize(w1), vectorize(w2));
  }

  public static void saveWordVectorSubset(Set<String> words, String allWordVectors,
      String saveLoc) throws IOException {
    IOs.writeLines(
        saveLoc,
        parseWordVectors(Paths.get(allWordVectors))
        // Aggressively normalizes phrases
            .filter(w -> words.contains(w.surface.replace('_', ' ').toLowerCase())).map(
                Phrase::toVectorStr));
  }

  public static String wordVectorStr(String w) {
    return w + " " + StringUtils.join(ArrayUtils.toObject(vectors.get(w)), ' ');
  }

  /**
   * Loads vectors into the Word class so newly created Words will have vector
   * representation from the loaded file
   * 
   * @param wordVectorFile
   * @return The loaded word to vector mapping
   * @throws IOException
   */
  public static Map<String, double[]> loadWordVectors(String wordVectorFile)
      throws IOException {
    Path filePath = Paths.get(wordVectorFile);
    try (Stream<Phrase> vs = parseWordVectors(filePath)) {
      vectors = vs.collect(Collectors.toMap(w -> w.surface, w -> w.cachedVector));
      lowercaseVectors = HashMultimap.create();
      for (String word : vectors.keySet()) {
        lowercaseVectors.put(word.toLowerCase(), word);
      }
      System.out.println("Loaded " + vectors.size() + " word vectors of dimension "
          + vectorDimension() + " from\n" + filePath.normalize());
      zeroVector = new double[vectorDimension()];
      return vectors;
    }
  }

  public static Stream<Phrase> parseWordVectors(Path filePath) throws IOException {
    return Files.lines(filePath).filter(s->s.length()>100).map(line -> {
      String[] parts = StringUtils.split(line);
      return new Phrase(parts[0].replace('_', ' '), parseVectors(parts));
    });
  }

  /**
   * Splits the phrase into words and average their output
   * 
   * @param phrase
   * @return averaged vector
   */
  public static double[] averagePhraseVector(String phrase) {
    return averagePhraseVector(Stream.of(StringUtils.split(phrase)));
  }

  /**
   * Averages the stream of words' vectors
   * 
   * @param phrase
   * @return averaged vector
   */
  public static double[] averagePhraseVector(Stream<String> words) {
    double[] vecSum = new double[Phrase.vectorDimension()];
    MutableInt vecCount = new MutableInt(0);
    // Prevent loss of precision
    words.map(Phrase::vectorize).forEach(v -> {
      if (v != null)
        ArrayMath.pairwiseAddInPlace(vecSum, v);
      vecCount.increment();
    });
    return ArrayMath.multiply(vecSum, 1 / vecCount.doubleValue());
  }

  /**
   * 
   * @return the current dimensions of word vectors used
   */
  public static int vectorDimension() {
    if (vectors == null || vectors.isEmpty())
      throw new IllegalStateException("No word vectors are loaded.");
    return vectors.values().iterator().next().length;
  }
  
  public static String stem(String word) {
    PorterStemmer stemmer = new PorterStemmer();
    stemmer.setCurrent(word);
    stemmer.stem();
    return stemmer.getCurrent();
  }
  
  public static String lowercasedStem(String word) {
    return stem(word.toLowerCase());
  }

}
