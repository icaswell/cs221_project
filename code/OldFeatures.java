package edu.stanford.nlp.semeval2015.pit;

import static edu.stanford.nlp.math.ArrayMath.dotProduct;
import static edu.stanford.nlp.math.ArrayMath.norm;
import it.unimi.dsi.fastutil.doubles.DoubleArrayList;

import java.util.AbstractList;
import java.util.Collection;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;

import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.Collections2;
import com.google.common.collect.Lists;
import com.google.common.collect.Multimap;
import com.google.common.collect.Sets;
import com.google.common.primitives.Doubles;

import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.math.ArrayMath;
import edu.stanford.nlp.semeval2015.pit.Phrase.WordPair;

class OldFeatures {
  
  static final int SIMILARITY_BIN_COUNT = 10;
  
  static DoubleArrayList highSimPair(ParaphraseProblem p){
    List<Phrase> ph1 = p.s1.getAllPhrases(3);
    List<Phrase> ph2 = p.s2.getAllPhrases(3);
  
    List<WordPair> alignSim = Lists.newArrayList();
    // Only look at valid words
    for (Phrase w1 : ph1) {
      for (Phrase w2 : ph2) {
          alignSim.add(new WordPair(w1, w2));
      }
    }
    
//    Stream<WordPair> goodPairs = alignSim.stream()
//        .filter(w->!w.areSameWord && w.score>0.0 && !w.hasStopWord())
//        .sorted();
//    
//    List<WordPair> pairList = goodPairs.collect(Collectors.toList());
//    System.out.println(pairList);
    Collection<WordPair> goodPairs = Collections2.filter(alignSim,
        w-> w.score>0.0 && !w.hasStopWord());
//    System.out.println(goodPairs);
//    System.out.println(p);
//    System.out.println();
    double[] simScores = goodPairs.stream()
        .mapToDouble(w->w.score)
        .toArray();
    
    if (simScores.length == 0)
      simScores = new double[] { 0 };
    
    double[] binningFeats = binning(simScores,SIMILARITY_BIN_COUNT);
    DoubleArrayList feats = new DoubleArrayList(binningFeats);
    
    feats.add( Doubles.max(simScores));
    feats.add( Doubles.min(simScores));
    feats.add( ArrayMath.average(simScores));
    feats.add( ArrayMath.sum(simScores));
    return feats;
  }
  
  /**
   * Describes a distribution using historgram
   * assume the raw numbers are normalized to between 0-1
   * @param rawNumbers between 0-1
   * @param bins number of bins to divide the scores into
   * @return the historgram for bins
   */
  public static double[] binning(double[] rawNumbers,int bins){
    double gap = 1.0/bins;
    double[] bucket = new double[bins+1];
    double normalizedIncrement = 1./rawNumbers.length;
    for (double d : rawNumbers) {
      int index = (int) Math.floor(d / gap);
      bucket[index] += normalizedIncrement;
    }
    for (int i = 0; i < bucket.length; i++) {
      bucket[i] = Math.log(bucket[i] + 0.000001);
    }
    return bucket;
  }
  
  private static Multimap<String,Integer> getWordIndices(List<String> tokens){
    Multimap<String,Integer> indices = ArrayListMultimap.create();
    for (int i = 0; i < tokens.size(); i++) {
      String word = tokens.get(i);
      indices.put(word, i);
    }
    return indices;
  }

  
  static void ruleAlign(ParaphraseProblem p){
    
    Sentence s1 = p.s1;
    List<CoreLabel> t1 = p.s1.getWords();
    Multimap<String,Integer> i1 = getWordIndices(p.s1.lowercaseTokens);
    
    Sentence s2 = p.s2;
    List<CoreLabel> t2 = p.s2.getWords();
    Multimap<String,Integer> i2 = getWordIndices(p.s2.lowercaseTokens);
    
    Set<String> commonPhrases = Sets.intersection(i1.keySet(), i2.keySet());

    // 0. Place "bars" in the sentence to prevent generation of 
    // bigrams etc at the boundary.
    
//    String label = "";
//    List<String> entity = Lists.newArrayList();
//    for(CoreLabel w:s1){
//      System.out.println(w+"\t"+w.ner());
//      if(label.equals(w.ner())){
//        entity.add(w.originalText().toLowerCase());
//      }
//    }
//    
//    for(CoreLabel w:s2){
//      System.out.println(w+"\t"+w.ner());
//    }
    
    LinkedList<String> words = Lists.newLinkedList(s1.tokens);
    // 1. Remove all identical 1+ ngrams from both sentences
    List<Integer> segments = Lists.newArrayList();
    for (int i = 0; i < s1.length(); i++) {
      if(!commonPhrases.contains(s1.tokens.get(i).toLowerCase())){
        segments.add(i);
      }
    }
    
    //    a. find longest common substring
//    LongestCommonSubsequence<String> lcs = new LongestCommonSubsequence<String>();
//    lcs.getLCSMatch(s1, s2);
    
    // 2. Find nearest neighbors on the remaining shingles
  }
  

  /**
   * Represents the abstract list of pairs in sentence pairs
   *
   */
  static class Alignment extends AbstractList<String[]>{

    // aligned segments
    private List<List<String>> s1;
    
    private List<List<String>> s2;
    
    private ParaphraseProblem p;
    
    private List<String[]> phrasePairs = Lists.newArrayList();
    
    public double maxScore(){
      return phrasePairs.stream().mapToDouble(p->{
        double[] v1 = Phrase.vectorize(p[0]);
        double[] v2 = Phrase.vectorize(p[1]);
        return dotProduct(v1, v2) / (norm(v1) * norm(v2));
      }).max().getAsDouble();
    }

    @Override
    public String[] get(int index) {
      return phrasePairs.get(index);
    }

    @Override
    public int size() {
      return phrasePairs.size();
    }
    
  }
}
